"""[harness addition] Shared machinery for our natural-gradient (trait-flow) steering in IDEEA's harness.

Intervention protocol (identical to IDEEA's CAA hook): theta is added to the output of decoder layer L
at the LAST position of every forward call during generation, i.e. at the last prompt token and at
every generated position. In a teacher-forced forward pass over prompt + continuation this equals
adding theta at positions P-1 .. P+T-2 (the positions that predict the T continuation tokens).

Score features u = grad_theta log p(y | x, theta) at theta = 0 are computed with that same protocol,
so G = E[u u^T] is the Fisher of the (sequence) readout of exactly the intervention we apply.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common.proxy_judge import EmbJudge, clean, split_refs  # noqa: E402

MODEL_ID = os.environ.get("NG_MODEL", "Qwen/Qwen3-1.7B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class RowSteer:
    """Adds theta (B, d) or (d,) to the output of one decoder layer.

    mode "last": add at the last position of every call (generation; IDEEA's protocol).
    mode "from": add at positions >= start (teacher-forced pass over prompt + continuation).
    """

    def __init__(self, model, layer):
        self.theta = None
        self.mode = "last"
        self.start = None
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.theta is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        th = self.theta.to(h.dtype)
        if th.dim() == 1:
            th = th[None]
        if self.mode == "last":
            add = torch.zeros_like(h)
            add[:, -1:] = th[:, None, :].expand(h.shape[0], 1, -1) if th.shape[0] == 1 else th[:, None, :]
        else:
            mask = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
            mask[self.start:] = 1
            add = mask[None, :, None] * th[:, None, :]
        h2 = h + add
        if isinstance(output, tuple):
            return (h2,) + tuple(output[1:])
        return h2

    def remove(self):
        self.handle.remove()


def load_model(dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.generation_config.pad_token_id = tok.pad_token_id
    return tok, model


def prompt_ids(tok, chat):
    """IDEEA's eval encoding: chat template, continue the final 'A: ' assistant message."""
    enc = tok.apply_chat_template(chat, continue_final_message=True, return_dict=True, return_tensors="pt")
    return enc["input_ids"][0]


def stop_ids(tok):
    ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


@torch.no_grad()
def sample(model, tok, pid, n, steer=None, theta=None, max_new=50, temperature=1.0, bs=64):
    """n continuations at temperature T (top_k=0, top_p=1). Each kept up to and including its stop token."""
    stops = stop_ids(tok)
    conts = []
    for b in range(0, n, bs):
        m = min(bs, n - b)
        if steer is not None:
            steer.mode, steer.theta = "last", theta
        out = model.generate(pid[None].repeat(m, 1).to(DEV), attention_mask=torch.ones(m, pid.numel(), dtype=torch.long, device=DEV),
                             do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                             max_new_tokens=max_new, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        if steer is not None:
            steer.theta = None
        for row in out[:, pid.numel():].tolist():
            stop = next((i for i, t in enumerate(row) if t in stops), len(row) - 1)
            conts.append(torch.tensor(row[: stop + 1]))
    return conts


def _prefix_cache(model, pid):
    """KV cache of prompt positions 0..P-2 (they do not depend on theta), as a list of (k, v) per layer."""
    with torch.no_grad():
        out = model(input_ids=pid[None, :-1].to(DEV), use_cache=True)
    pkv = out.past_key_values
    kv = []
    for l in range(len(model.model.layers)):
        lay = pkv.layers[l] if hasattr(pkv, "layers") else None
        k, v = (lay.keys, lay.values) if lay is not None else pkv[l]
        kv.append((k, v))
    return kv


def _expand_cache(model, kv, B):
    from transformers import DynamicCache
    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()
    for l, (k, v) in enumerate(kv):
        cache.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), l)
    return cache


def cont_logits(model, steer, pid, conts, theta, kv=None):
    """Logits predicting each continuation token, with theta (B,d)/(d,)/None added at positions >= P-1.
    Uses a no-grad KV cache for prompt positions 0..P-2 (theta never touches them), then runs
    [last prompt token] + continuation[:-1] with the hook on every position."""
    P = pid.numel()
    if kv is None:
        kv = _prefix_cache(model, pid)
    B = len(conts)
    T = max(c.numel() for c in conts)
    ids = torch.full((B, T), 0, dtype=torch.long, device=DEV)
    tgt = torch.full((B, T), 0, dtype=torch.long, device=DEV)
    tmask = torch.zeros(B, T, device=DEV)
    for i, c in enumerate(conts):
        c = c.to(DEV)
        seq = torch.cat([pid[-1:].to(DEV), c[:-1]])
        ids[i, : seq.numel()] = seq
        tgt[i, : c.numel()] = c
        tmask[i, : c.numel()] = 1
    att = torch.cat([torch.ones(B, P - 1, dtype=torch.long, device=DEV), (tmask > 0).long()], 1)
    pos = torch.arange(P - 1, P - 1 + T, device=DEV)[None].expand(B, -1)
    steer.mode, steer.start, steer.theta = "from", 0, theta
    try:
        logits = model(input_ids=ids, attention_mask=att, position_ids=pos,
                       past_key_values=_expand_cache(model, kv, B), use_cache=True).logits.float()
    finally:
        steer.theta = None
    return logits, tgt, tmask


def score_features(model, steer, pid, conts, d, bs=16):
    """u_k = grad_theta log p(y_k | x, theta)|_0, theta at positions P-1.. (IDEEA's protocol). Returns (n, d) float32
    and the per-sample sequence log-prob."""
    U, LP = [], []
    kv = _prefix_cache(model, pid)
    for b in range(0, len(conts), bs):
        cs = conts[b:b + bs]
        theta = torch.zeros(len(cs), d, device=DEV, dtype=torch.float32, requires_grad=True)
        logits, tgt, tmask = cont_logits(model, steer, pid, cs, theta, kv)
        lp = (torch.log_softmax(logits, -1).gather(-1, tgt[..., None])[..., 0] * tmask).sum(1)
        lp.sum().backward()
        U.append(theta.grad.detach().cpu().numpy().astype(np.float32))
        LP.append(lp.detach().cpu().numpy())
        del logits, theta
    return np.concatenate(U), np.concatenate(LP)


@torch.no_grad()
def seq_kl(model, steer, pid, conts, theta, bs=32):
    """Rao-Blackwellised sequence KL(p_theta || p_0) for y ~ p_theta: sum_t KL(p_theta(.|y<t) || p_0(.|y<t)),
    plus the plain Monte-Carlo log-ratio. Returns per-sample arrays (rb, mc)."""
    rb, mc = [], []
    kv = _prefix_cache(model, pid)
    for b in range(0, len(conts), bs):
        cs = conts[b:b + bs]
        lt, tgt, tmask = cont_logits(model, steer, pid, cs, theta, kv)
        l0, _, _ = cont_logits(model, steer, pid, cs, None, kv)
        lpt, lp0 = torch.log_softmax(lt, -1), torch.log_softmax(l0, -1)
        kl_tok = (lpt.exp() * (lpt - lp0)).sum(-1)
        rb.append((kl_tok * tmask).sum(1).cpu().numpy())
        r = (lpt.gather(-1, tgt[..., None]) - lp0.gather(-1, tgt[..., None]))[..., 0]
        mc.append((r * tmask).sum(1).cpu().numpy())
    return np.concatenate(rb), np.concatenate(mc)


def ridge_solve(G, g, lam):
    d = G.shape[0]
    return np.linalg.solve(G + lam * np.eye(d), g)

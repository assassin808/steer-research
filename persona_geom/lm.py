"""Gemma-2-2B-it wrapper: chat formatting, steering hooks, sampling, teacher-forced response-mean residuals, per-token KL.
Device-agnostic (cuda bf16 if available, else cpu fp32) so every stage can be smoke-tested on the login node.

Steering modes at one layer L (residual stream = output of decoder layer L, i.e. hidden_states[L+1]):
- "add": h += delta_row at ALL positions (prompt and generated), as in persona_vectors / CAA-style steering.
- "ideea": IDEEA-CAA residual rule (~/ext/IDEEA/methods/ideea_caa.py, variant nearest_pos_neg): at the LAST position of every
  forward call, x <- x + alpha * (nearest pos centroid - nearest neg centroid), selected from x itself (float64).
  In a teacher-forced pass the same rule is applied at every position >= last prompt token. This is exactly equivalent:
  the un-edited layer-L output at position t depends only on layers <= L at positions <= t, and the edits enter only
  layers > L, so the selected direction is the same as during generation. (Checked numerically in smoke_test.py.)
"""
import os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import GEN_MODEL, log

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
if DEV == "cpu":
    torch.set_num_threads(int(os.environ.get("NT", "16")))


class LM:
    def __init__(self, name=GEN_MODEL):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=DT, attn_implementation="eager").to(DEV).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.layers = self.model.model.layers
        self.nl, self.H = len(self.layers), self.model.config.hidden_size
        self.pad = self.tok.pad_token_id
        eot = self.tok.convert_tokens_to_ids("<end_of_turn>")
        self.stop = sorted({eot, self.tok.eos_token_id})
        self.hook = dict(L=None, mode=None, delta=None, ideea=None, from_pos=None)
        for li, layer in enumerate(self.layers):
            layer.register_forward_hook(self._mk(li))

    # ---------------- hooks ----------------
    def _mk(self, li):
        def f(mod, inp, out):
            hk = self.hook
            if hk["L"] != li:
                return out
            h = out[0] if isinstance(out, tuple) else out
            if hk["mode"] == "add":
                h = h + hk["delta"].to(h.dtype)
            elif hk["mode"] == "ideea":
                pc, nc, alpha = hk["ideea"]["pos"], hk["ideea"]["neg"], hk["ideea"]["alpha"]  # [K,H] f64, [K,H] f64, [n] f64
                h = h.clone()
                if hk["from_pos"] is None:              # generation: last position of this forward call
                    x = h[:, -1].to(torch.float64)
                    h[:, -1] = (x + alpha[:, None] * _npn(x, pc, nc)).to(h.dtype)
                else:                                   # teacher forcing: every position >= from_pos[row]
                    n, S, _ = h.shape
                    x = h.to(torch.float64)
                    d = _npn(x.reshape(n * S, -1), pc, nc).reshape(n, S, -1)
                    m = (torch.arange(S, device=h.device)[None] >= hk["from_pos"][:, None]).to(torch.float64)
                    h = (x + alpha[:, None, None] * m[..., None] * d).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    def set_add(self, L, deltas):
        """deltas: list (len n) of np [H] or None"""
        if L is None or all(d is None for d in deltas):
            self.clear(); return
        D = np.zeros((len(deltas), self.H), np.float32)
        for j, d in enumerate(deltas):
            if d is not None: D[j] = d
        self.hook.update(L=L, mode="add", delta=torch.tensor(D, device=DEV)[:, None, :], ideea=None, from_pos=None)

    def set_ideea(self, L, pos_c, neg_c, alphas, from_pos=None):
        self.hook.update(L=L, mode="ideea", delta=None, from_pos=from_pos,
                         ideea=dict(pos=torch.tensor(pos_c, dtype=torch.float64, device=DEV),
                                    neg=torch.tensor(neg_c, dtype=torch.float64, device=DEV),
                                    alpha=torch.tensor(np.asarray(alphas, np.float64), device=DEV)))

    def clear(self):
        self.hook.update(L=None, mode=None, delta=None, ideea=None, from_pos=None)

    # ---------------- formatting ----------------
    def chat_ids(self, system, q):
        """Gemma-2 has no system role: the system text is prepended to the user turn (standard practice for Gemma)."""
        content = q if not system else f"{system}\n\n{q}"
        s = self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        return self.tok(s, add_special_tokens=False).input_ids

    def decode(self, r):
        return self.tok.decode([t for t in r if t not in self.stop], skip_special_tokens=True).strip()

    # ---------------- sampling ----------------
    @torch.no_grad()
    def generate(self, ctxs, seed, maxnew, steer=None, bs=None):
        """ctxs: list of token-id lists. steer(chunk_idx_list) sets hooks for that chunk (or None).
        T=1, top_p=1, top_k off. Returns list of generated id lists (stop token kept if produced)."""
        bs = bs or (64 if DEV == "cuda" else 8)
        order = sorted(range(len(ctxs)), key=lambda i: -len(ctxs[i]))
        res = [None] * len(ctxs)
        for b in range(0, len(order), bs):
            ch = order[b:b + bs]
            S = max(len(ctxs[i]) for i in ch)
            ids = torch.full((len(ch), S), self.pad, dtype=torch.long); mask = torch.zeros_like(ids)
            for j, i in enumerate(ch):
                ids[j, S - len(ctxs[i]):] = torch.tensor(ctxs[i]); mask[j, S - len(ctxs[i]):] = 1
            if steer is not None: steer(ch)
            torch.manual_seed(seed + b)
            out = self.model.generate(input_ids=ids.to(DEV), attention_mask=mask.to(DEV), do_sample=True, temperature=1.0,
                                      top_p=1.0, top_k=0, max_new_tokens=maxnew, eos_token_id=self.stop, pad_token_id=self.pad)
            self.clear()
            for j, i in enumerate(ch):
                r = out[j, S:].tolist()
                cut = next((k for k, t in enumerate(r) if t in self.stop), None)
                res[i] = r[:cut + 1] if cut is not None else r
        return res

    # ---------------- teacher forcing ----------------
    @torch.no_grad()
    def tf(self, ctxs, resps, layers=None, want_lp=False, steer=None, bs=None):
        """Teacher-force ctx+resp. Returns (acts [n, len(layers), H] float16: mean residual over RESPONSE tokens, stop token
        excluded, hidden_states[l] for l in layers (0 = embeddings, l = output of block l-1); logprobs list of [T_i, V] or None)."""
        bs = bs or (16 if DEV == "cuda" else 4)
        n = len(ctxs); acts = None; lps = [None] * n
        for b in range(0, n, bs):
            ch = list(range(b, min(n, b + bs)))
            seqs = [ctxs[i] + resps[i] for i in ch]
            S = max(len(s) for s in seqs)
            ids = torch.full((len(ch), S), self.pad, dtype=torch.long); mask = torch.zeros_like(ids)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s); mask[j, :len(s)] = 1
            if steer is not None: steer(ch)
            o = self.model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV), output_hidden_states=layers is not None)
            self.clear()
            if layers is not None:
                if acts is None: acts = np.zeros((n, len(layers), self.H), np.float16)
                T = S
                m = torch.zeros((len(ch), T), device=DEV)
                for j, i in enumerate(ch):
                    r = [t for t in resps[i] if t not in self.stop]
                    m[j, len(ctxs[i]):len(ctxs[i]) + max(1, len(r))] = 1.0
                den = m.sum(1, keepdim=True).clamp(min=1)
                for k, l in enumerate(layers):
                    a = o.hidden_states[l].float()
                    acts[ch, k] = ((a * m[..., None]).sum(1) / den).cpu().numpy().astype(np.float16)
            if want_lp:
                lp = torch.log_softmax(o.logits.float(), -1)
                for j, i in enumerate(ch):
                    c, T = len(ctxs[i]), len(resps[i])
                    lps[i] = lp[j, c - 1:c - 1 + T]
            del o
        return acts, (lps if want_lp else None)


def _npn(x, pc, nc):
    """IDEEA-CAA nearest_pos_neg: nearest pos centroid minus nearest neg centroid (independently), not normalised."""
    dp = ((x[:, None] - pc[None]) ** 2).sum(-1); dn = ((x[:, None] - nc[None]) ** 2).sum(-1)
    return pc[dp.argmin(-1)] - nc[dn.argmin(-1)]


def kl_per_token(lpA, lpB, resps):
    """mean over response tokens of KL(pA || pB), exact over vocab"""
    out = []
    for a, b, r in zip(lpA, lpB, resps):
        if len(r) == 0: out.append(0.0); continue
        out.append(float((a.exp() * (a - b)).sum(-1).mean()))
    return out

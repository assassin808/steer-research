"""[harness addition] Track 1 evaluation: IDEEA's TruthfulQA test split, greedy, 50 new tokens (IDEEA's protocol),
for IDEEA's methods (installed through IDEEA's own classes) and ours (residual stream at layer L, IDEEA's CAA
protocol), with realized KL and the proxy judges.

Spec strings (method:param[:seed]):
  base
  iti:ALPHA            IDEEA ITI, k=48 heads (IDEEA default k)
  ideea:ALPHA          IDEEA nearest_pos_neg, nc=4, k=48 (IDEEA defaults)
  caa:C                CAA mean-diff direction at L, unit-normalised, norm C * hnorm
  ng:C                 our global natural-gradient direction (G+lam I)^-1 g, norm C * hnorm
  euc:C                plain readout-consistent trait gradient g (no metric), norm C * hnorm
  ngE:C / eucE:C       same, fitted on embedding-judge scores instead of Qwen-judge scores
  rand:C:SEED          random Gaussian direction, norm C * hnorm
  ngkl:EPS             global ng direction, per-input norm s.t. 1/2 v^T G_x v = EPS (G_x: local sequence Fisher)
  ngx:EPS              per-input direction (G_x + lam I)^-1 g, per-input norm s.t. 1/2 v^T G_x v = EPS
  caakl:EPS            CAA direction with the same per-input KL calibration (control for ngkl)
hnorm = mean L2 norm of the layer-L residual at the last token over IDEEA's contrast prompts.
G_x = 0.5 * (within-sample Fisher from 16 unlabelled base samples at x) + 0.5 * global G (fixed shrinkage, not tuned).

Realized KL: mean over generated tokens of KL(p_steered(.|x,y<t) || p_base(.|x,y<t)) along the steered greedy
response (steered logits from generate, base logits from a teacher-forced pass with hooks removed); also first-token KL.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng_lib as L  # noqa: E402
from common.config import deterministic  # noqa: E402
from common.paths import activations_path, dataset_dir  # noqa: E402
from common.proxy_judge import QwenJudge  # noqa: E402
from methods.ideea import IDEEA, FixedClusters  # noqa: E402
from methods.iti import ITI  # noqa: E402
from tasks.truthfulqa import TruthfulQA  # noqa: E402
from common.io import load_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs_ng")
MODEL_NAME = "qwen3_1.7b"


class Ctx:
    def __init__(self, layer):
        self.layer = layer
        self.task = TruthfulQA()
        self.df = self.task._load()
        self.requests = self.task.eval_requests()
        # primary: directions fitted on Qwen-judge scores; "E" variants: fitted on embedding-judge scores
        D = np.load(f"{RUNS}/directions_L{layer}_qwen.npz")
        self.ng, self.g, self.G = D["ng"], D["g"], D["G"]
        self.lam = json.load(open(f"{RUNS}/fit_L{layer}_qwen.json"))["lam_rel"] * np.trace(self.G) / self.G.shape[0]
        DE = np.load(f"{RUNS}/directions_L{layer}_emb.npz")
        self.ngE, self.gE = DE["ng"], DE["g"]
        caa = load_json(os.path.join(ROOT, f"results/truthfulqa/{MODEL_NAME}/directions/caa/directions.json"))
        self.caa = np.array(caa["direction"][layer])
        acts = pd.read_pickle(activations_path("truthfulqa", MODEL_NAME, "residual"))
        self.hnorm = float(np.mean([np.linalg.norm(v) for v in acts[f"layer_{layer}"]]))
        self._Gx = None

    def Gx(self, key):
        """Blended local metric at test question `key` (unlabelled samples only)."""
        if self._Gx is None:
            import glob
            fs = sorted(glob.glob(f"{RUNS}/test_L{self.layer}_shard*.npz"))
            self._Gx = {}
            for f in fs:
                z = np.load(f)
                U, q = z["U"].astype(np.float64), z["qidx"]
                for i in np.unique(q):
                    Ui = U[q == i] - U[q == i].mean(0)
                    self._Gx[int(i)] = Ui
        Ui = self._Gx[int(key)]
        return 0.5 * (Ui.T @ Ui / len(Ui)) + 0.5 * self.G


def unit(v):
    return v / np.linalg.norm(v)


def per_input_thetas(ctx, kind, eps, keys):
    out, pred = [], []
    for k in keys:
        Gx = ctx.Gx(k)
        if kind == "ngx":
            v = np.linalg.solve(Gx + ctx.lam * np.eye(len(ctx.g)), ctx.g)
        elif kind == "ngkl":
            v = ctx.ng
        else:
            v = ctx.caa
        q = float(v @ Gx @ v)
        out.append(v * np.sqrt(2 * eps / q))
        pred.append(eps)
    return np.stack(out)


def build(ctx, model, spec):
    """Returns (installer(batch_keys) -> callable to clear, info, steer_obj_or_None)."""
    parts = spec.split(":")
    m = parts[0]
    if m == "base":
        return None, {"method": "base"}
    val = float(parts[1])
    if m in ("iti", "ideea"):
        meth = ITI(48, val) if m == "iti" else IDEEA("nearest_pos_neg", FixedClusters(4), 48, val)
        return ("ideea_hooks", meth), {"method": m, "alpha": val, "k": 48}
    if m in ("caa", "ng", "euc", "rand", "ngE", "eucE"):
        if m == "rand":
            seed = int(parts[2]) if len(parts) > 2 else 0
            v = np.random.RandomState(1000 + seed).randn(len(ctx.g))
        else:
            v = {"caa": ctx.caa, "ng": ctx.ng, "euc": ctx.g, "ngE": ctx.ngE, "eucE": ctx.gE}[m]
        th = unit(v) * val * ctx.hnorm
        return ("fixed", th), {"method": m, "c": val, "norm": float(np.linalg.norm(th)),
                               "pred_kl_globalG": float(0.5 * th @ ctx.G @ th)}
    if m in ("ngkl", "ngx", "caakl"):
        return ("perinput", m, val), {"method": m, "eps": val}
    raise ValueError(spec)


def run_spec(ctx, model, tok, steer, spec, bs, out_dir):
    path = os.path.join(out_dir, spec.replace(":", "_") + ".json")
    if os.path.exists(path):
        print("exists", path)
        return
    deterministic(0)
    plan, info = build(ctx, model, spec)
    reqs = ctx.requests
    gen = L  # noqa
    from common.modeling import LocalGenerator
    lg = LocalGenerator(model, tok, True, bs, ctx.task.max_new_tokens, None)
    stops = L.stop_ids(tok)
    records = []
    hooks = None
    if plan is not None and plan[0] == "ideea_hooks":
        hooks = plan[1].install(model, ctx.task, MODEL_NAME)
    t0 = time.time()
    norms = []
    try:
        for b in range(0, len(reqs), bs):
            batch = reqs[b:b + bs]
            enc = lg._encode([r.chat for r in batch]).to("cuda")
            P = enc["input_ids"].shape[1]
            theta = None
            if plan is not None and plan[0] == "fixed":
                theta = torch.tensor(plan[1], dtype=torch.float32, device="cuda")
            elif plan is not None and plan[0] == "perinput":
                th = per_input_thetas(ctx, plan[1], plan[2], [r.key for r in batch])
                norms += list(np.linalg.norm(th, axis=1))
                theta = torch.tensor(th, dtype=torch.float32, device="cuda")
            steer.mode, steer.theta = "last", theta
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=ctx.task.max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, output_logits=True, return_dict_in_generate=True)
            steer.theta = None
            seq = out.sequences
            gen_ids = seq[:, P:]
            Tg = gen_ids.shape[1]
            # length up to and including the first stop token
            lens = []
            for row in gen_ids.tolist():
                lens.append(next((i + 1 for i, t in enumerate(row) if t in stops), len(row)))
            lens = torch.tensor(lens, device="cuda")
            lst = torch.stack(out.logits, 1).float()  # (B, Tg, V) steered
            # base teacher-forced pass, all hooks off
            if hooks is not None:
                hooks.remove()
            att = torch.cat([enc["attention_mask"], (torch.arange(Tg, device="cuda")[None] < lens[:, None]).long()], 1)
            with torch.no_grad():
                lb = model(input_ids=seq, attention_mask=att).logits[:, P - 1:P - 1 + Tg].float()
            if hooks is not None:
                hooks = plan[1].install(model, ctx.task, MODEL_NAME)
            ls_, lb_ = torch.log_softmax(lst, -1), torch.log_softmax(lb, -1)
            kl = (ls_.exp() * (ls_ - lb_)).sum(-1)  # (B, Tg)
            tm = (torch.arange(Tg, device="cuda")[None] < lens[:, None]).float()
            kl_mean = ((kl * tm).sum(1) / tm.sum(1)).cpu().numpy()
            kl_first = kl[:, 0].cpu().numpy()
            texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
            for r, t, km, kf, n in zip(batch, texts, kl_mean, kl_first, lens.tolist()):
                records.append({"key": r.key, "meta": r.meta, "response": ctx.task.parse(t), "kl_tok": float(km),
                                "kl_first": float(kf), "ntok": int(n)})
            del lst, lb, ls_, lb_, out
    finally:
        if hooks is not None:
            hooks.remove()
        steer.theta = None
    gen_time = time.time() - t0
    # judges (hooks are off)
    res = ctx.task._proxy_judge(records, "emb")
    qj = QwenJudge(model=model, tokenizer=tok)
    items = []
    for r in records:
        i = int(r["key"])
        items.append((r["meta"]["question"], res["judgements"][str(r["key"])]["answer_clean"] or "(empty)",
                      L.split_refs(ctx.df.loc[i, "Correct Answers"]), L.split_refs(ctx.df.loc[i, "Incorrect Answers"])))
    py = qj.p_yes(items)
    for r, p in zip(records, py):
        j = res["judgements"][str(r["key"])]
        r.update({"answer_clean": j["answer_clean"], "truth_emb": j["truth"], "info": j["info"], "score_emb": j["score_emb"],
                  "p_yes_qwen": float(p), "truth_qwen": bool(j["info"] is False or p > qj.thr)})
    te = np.array([r["truth_emb"] for r in records]); tq = np.array([r["truth_qwen"] for r in records])
    inf = np.array([r["info"] for r in records]); kl = np.array([r["kl_tok"] for r in records])
    summ = {"spec": spec, **info, "n": len(records), "gen_seconds": gen_time,
            "truth_emb": te.mean(), "truth_qwen": tq.mean(), "info": inf.mean(),
            "ti_emb": (te & inf).mean(), "ti_qwen": (tq & inf).mean(),
            "kl_tok_mean": kl.mean(), "kl_tok_median": float(np.median(kl)), "kl_first_mean": float(np.mean([r["kl_first"] for r in records])),
            "mean_ntok": float(np.mean([r["ntok"] for r in records]))}
    if norms:
        summ["mean_norm"] = float(np.mean(norms)); summ["sd_norm"] = float(np.std(norms))
    summ = {k: (float(v) if isinstance(v, (np.floating, np.bool_)) else v) for k, v in summ.items()}
    json.dump({"summary": summ, "records": records}, open(path, "w"))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default="grid")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "results_t1", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    ctx = Ctx(a.layer)
    tok, model = L.load_model(torch.bfloat16)  # IDEEA evaluates in bf16
    steer = L.RowSteer(model, a.layer)
    print(f"hnorm(L{a.layer}) = {ctx.hnorm:.2f}; |caa| = {np.linalg.norm(ctx.caa):.2f}; lam = {ctx.lam:.3g}", flush=True)
    for s in a.specs:
        run_spec(ctx, model, tok, steer, s, a.bs, out_dir)

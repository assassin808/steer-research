"""[harness addition] Gemma-2-2B-it unified evaluation: generate on IDEEA's TruthfulQA test split (greedy, 50 new
tokens, bf16, IDEEA prompt) for IDEEA/ITI/CAA (IDEEA's own method classes) and our directions, with realized
per-token KL(steered || base) along the steered answer. Judging (official judges) is a separate bulk step (gm_judge.py).

Spec strings (method:params):
  base
  iti:ALPHA                 IDEEA ITI, k=48 (IDEEA default k)
  ideea:ALPHA               IDEEA nearest_pos_neg, nc=4, k=48 (IDEEA defaults)
  caa:LAYER:ALPHA           IDEEA CAA class (direction * alpha, NOT normalised)
  cu:LAYER:C                CAA mean-diff at LAYER, unit-normalised, norm C*hnorm(LAYER)
  ngref:C / eucref:C        judge-free (G+lam I)^-1 g_ref / g_ref at the fitted layer, norm C*hnorm
  ng:C / euc:C              judge-supervised variant (official-judge labels on TRAIN samples)
  rand:C:SEED               random Gaussian direction, norm C*hnorm
hnorm(L) = mean L2 norm of the layer-L residual at the last token over IDEEA's contrast prompts (residual.pkl).
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ng_lib as L  # noqa
from common.config import deterministic  # noqa
from common.io import load_json  # noqa
from common.paths import activations_path  # noqa
from common.modeling import LocalGenerator  # noqa
from methods.ideea import IDEEA, FixedClusters  # noqa
from methods.iti import ITI  # noqa
from methods.caa import CAA  # noqa
from tasks.truthfulqa import TruthfulQA  # noqa

ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, os.environ.get("NG_RUNS", "runs_ng"))
MODEL_NAME = "gemma2_2b"
FIT_LAYER = int(os.environ.get("FIT_LAYER", 13))


class Ctx:
    def __init__(self, split="test"):
        self.task = TruthfulQA(); self.requests = self.task.eval_requests()
        if split != "test":  # [harness] development split (selection of strength / layer; never test)
            from common.io import load_json as _lj
            from common.paths import dataset_dir
            from common.modeling import GenRequest
            from tasks.truthfulqa import _question_prompt
            df = self.task._load(); ids = _lj(dataset_dir("truthfulqa") / "split.json")[split]
            self.requests = [GenRequest(key=int(i), chat=[{"role": "user", "content": _question_prompt(df.loc[i, "Question"])}, {"role": "assistant", "content": "A: "}],
                                        meta={"question": df.loc[i, "Question"]}) for i in ids]
        self.acts = pd.read_pickle(activations_path("truthfulqa", MODEL_NAME, "residual"))
        self.caa = np.array(load_json(os.path.join(ROOT, f"results/truthfulqa/{MODEL_NAME}/directions/caa/directions.json"))["direction"])
        self._hn = {}
        self.dirs = {}
        for f, keys in ((f"directions_ref_L{FIT_LAYER}.npz", ("ngref", "eucref")), (f"directions_judged_L{FIT_LAYER}.npz", ("ng", "euc"))):
            if os.path.exists(f"{RUNS}/{f}"):
                z = np.load(f"{RUNS}/{f}")
                for k in keys:
                    self.dirs[k] = z[k]

    def hnorm(self, layer):
        if layer not in self._hn:
            self._hn[layer] = float(np.mean([np.linalg.norm(v) for v in self.acts[f"layer_{layer}"]]))
        return self._hn[layer]


def unit(v):
    return v / np.linalg.norm(v)


def build(ctx, spec):
    """-> (plan, info). plan: None | ("ideea_hooks", method) | ("fixed", theta, layer)."""
    p = spec.split(":"); m = p[0]
    if m == "base":
        return None, {"method": "base"}
    if m in ("iti", "ideea"):
        a = float(p[1])
        # iti:A[:K]   ideea:A[:K[:VARIANT[:NC]]]   (defaults: K=48, nearest_pos_neg, nc=4)
        k = int(p[2]) if len(p) > 2 else 48
        var = p[3] if len(p) > 3 else "nearest_pos_neg"; nc = int(p[4]) if len(p) > 4 else 4
        meth = ITI(k, a) if m == "iti" else IDEEA(var, FixedClusters(nc), k, a)
        return ("ideea_hooks", meth), {"method": m, "alpha": a, "k": k, "variant": var if m == "ideea" else None, "nc": nc if m == "ideea" else None}
    if m == "caa":
        layer, a = int(p[1]), float(p[2])
        return ("ideea_hooks", CAA(layer, a)), {"method": "caa", "layer": layer, "alpha": a, "direction_norm": float(np.linalg.norm(ctx.caa[layer]))}
    if m == "cu":
        layer, c = int(p[1]), float(p[2])
        th = unit(ctx.caa[layer]) * c * ctx.hnorm(layer)
        return ("fixed", th, layer), {"method": "cu", "layer": layer, "c": c, "norm": float(np.linalg.norm(th))}
    if m in ("ngref", "eucref", "ng", "euc", "rand"):
        c = float(p[1]); layer = FIT_LAYER
        v = np.random.RandomState(1000 + int(p[2])).randn(ctx.caa.shape[1]) if m == "rand" else ctx.dirs[m]
        th = unit(v) * c * ctx.hnorm(layer)
        return ("fixed", th, layer), {"method": m, "layer": layer, "c": c, "norm": float(np.linalg.norm(th))}
    raise ValueError(spec)


def run_spec(ctx, model, tok, steers, spec, bs, out_dir):
    path = os.path.join(out_dir, spec.replace(":", "_") + ".json")
    if os.path.exists(path):
        print("exists", path); return
    deterministic(0)
    plan, info = build(ctx, spec)
    lg = LocalGenerator(model, tok, True, bs, ctx.task.max_new_tokens, None)
    stops = L.stop_ids(tok)
    hooks = plan[1].install(model, ctx.task, MODEL_NAME) if plan and plan[0] == "ideea_hooks" else None
    steer = None
    if plan and plan[0] == "fixed":
        steer = steers.setdefault(plan[2], L.RowSteer(model, plan[2]))
    records, t0 = [], time.time()
    reqs = ctx.requests
    try:
        for b in range(0, len(reqs), bs):
            batch = reqs[b:b + bs]
            enc = lg._encode([r.chat for r in batch]).to("cuda")
            P = enc["input_ids"].shape[1]
            if steer is not None:
                steer.mode, steer.theta = "last", torch.tensor(plan[1], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=ctx.task.max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, output_logits=True, return_dict_in_generate=True)
            if steer is not None:
                steer.theta = None
            seq = out.sequences; gen_ids = seq[:, P:]; Tg = gen_ids.shape[1]
            lens = torch.tensor([next((i + 1 for i, t in enumerate(row) if t in stops), len(row)) for row in gen_ids.tolist()], device="cuda")
            lst = torch.stack(out.logits, 1)  # (B,Tg,V) steered
            kl_mean = kl_first = None
            if plan is None:
                klm = np.zeros(len(batch)); klf = np.zeros(len(batch))
            else:
                if hooks is not None:
                    hooks.remove()
                att = torch.cat([enc["attention_mask"], (torch.arange(Tg, device="cuda")[None] < lens[:, None]).long()], 1)
                with torch.no_grad():
                    lb = model(input_ids=seq, attention_mask=att).logits[:, P - 1:P - 1 + Tg]
                if hooks is not None:
                    hooks = plan[1].install(model, ctx.task, MODEL_NAME)
                kl = torch.zeros(len(batch), Tg, device="cuda")
                for t in range(Tg):
                    ls_, lb_ = torch.log_softmax(lst[:, t].float(), -1), torch.log_softmax(lb[:, t].float(), -1)
                    kl[:, t] = (ls_.exp() * (ls_ - lb_)).sum(-1)
                tm = (torch.arange(Tg, device="cuda")[None] < lens[:, None]).float()
                klm = ((kl * tm).sum(1) / tm.sum(1)).cpu().numpy(); klf = kl[:, 0].cpu().numpy()
                del lb
            texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
            for r, t, km, kf, n in zip(batch, texts, klm, klf, lens.tolist()):
                records.append({"key": r.key, "meta": r.meta, "response": ctx.task.parse(t), "kl_tok": float(km), "kl_first": float(kf), "ntok": int(n)})
            del lst, out
    finally:
        if hooks is not None:
            hooks.remove()
        if steer is not None:
            steer.theta = None
    kl = np.array([r["kl_tok"] for r in records])
    summ = {"spec": spec, **info, "n": len(records), "gen_seconds": time.time() - t0, "kl_tok_mean": float(kl.mean()),
            "kl_first_mean": float(np.mean([r["kl_first"] for r in records])), "mean_ntok": float(np.mean([r["ntok"] for r in records]))}
    json.dump({"summary": summ, "records": records}, open(path, "w"))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", default="grid"); ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "results_gm", a.split + "_" + a.tag); os.makedirs(out_dir, exist_ok=True)
    ctx = Ctx(a.split)
    tok, model = L.load_model(torch.bfloat16)
    steers = {}
    for s in a.specs:
        run_spec(ctx, model, tok, steers, s, a.bs, out_dir)

"""[harness addition] Fit our global natural-gradient direction on IDEEA's TruthfulQA train split.

  sample : for each question of a split, draw n continuations of IDEEA's eval prompt at T=1, judge them with the
           embedding proxy (s = truth AND info), compute score features u = grad_theta log p(y|x) at layer L
           (IDEEA's CAA intervention protocol, see ng_lib). Writes runs_ng/{split}_shardK.npz.
  fit    : g = mean_q Cov_q(s, u) (readout-consistent trait gradient), G = mean_q Cov_q(u) (sequence Fisher,
           within-question centred), v_ng = (G + lam I)^{-1} g with lam chosen by held-out R^2 on the val split,
           v_euc = g. Writes runs_ng/directions_L{L}.npz and a small json summary.
The test split is also sampled (unlabelled use only: its u's give the local metric G_x for the per-input variant;
its judge scores are used ONLY to pick questions for track 2, never by a steering method).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng_lib as L  # noqa: E402
from common.io import load_json  # noqa: E402
from common.paths import dataset_dir  # noqa: E402
from tasks.truthfulqa import _question_prompt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs_ng")


def split_ids(name):
    sp = load_json(dataset_dir("truthfulqa") / "split.json")
    return [int(i) for i in sp[name]]


def chat_for(q):
    return [{"role": "user", "content": _question_prompt(q)}, {"role": "assistant", "content": "A: "}]


def stage_sample(a):
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
    tok, model = L.load_model(torch.float32)
    steer = L.RowSteer(model, a.layer)
    judge = L.EmbJudge()
    d = model.config.hidden_size
    ids = split_ids(a.split)
    if a.max_q:
        ids = ids[: a.max_q]
    torch.manual_seed(a.seed)
    for k0 in range(0, len(ids), a.shard):
        path = f"{OUT}/{a.split}_L{a.layer}_shard{k0 // a.shard:03d}.npz"
        if os.path.exists(path):
            continue
        rec = {"U": [], "s": [], "truth": [], "info": [], "margin": [], "lp": [], "qidx": [], "texts": [], "ntok": []}
        t0 = time.time()
        for i in ids[k0:k0 + a.shard]:
            q = df.loc[i, "Question"]
            pid = L.prompt_ids(tok, chat_for(q))
            conts = L.sample(model, tok, pid, a.n)
            texts = [tok.decode(c, skip_special_tokens=True) for c in conts]
            t, f, m = judge.judge(q, texts, L.split_refs(df.loc[i, "Correct Answers"]), L.split_refs(df.loc[i, "Incorrect Answers"]))
            U, lp = L.score_features(model, steer, pid.cuda(), [c.cuda() for c in conts], d)
            rec["U"].append(U); rec["s"].append((t & f).astype(np.float32)); rec["truth"].append(t); rec["info"].append(f)
            rec["margin"].append(m); rec["lp"].append(lp); rec["qidx"].append(np.full(a.n, i)); rec["texts"] += texts
            rec["ntok"].append(np.array([c.numel() for c in conts]))
        np.savez(path, **{k: (np.concatenate(v) if k != "texts" else np.array(v)) for k, v in rec.items()})
        print(f"{path}: {len(ids[k0:k0 + a.shard])} q in {time.time() - t0:.0f}s; mean s {np.concatenate(rec['s']).mean():.3f}", flush=True)


def load_split(split, layer, judge="emb"):
    fs = sorted(glob.glob(f"{OUT}/{split}_L{layer}_shard*.npz"))
    zs = [np.load(f) for f in fs]
    D = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    if judge == "qwen":
        from common.proxy_judge import qwen_threshold
        py = np.concatenate([np.load(f.replace(".npz", "_qwen.npy")) for f in fs])
        D["s_emb"] = D["s"]
        D["s"] = (D["info"] & (py > qwen_threshold())).astype(np.float32)
    return D


def within_stats(D):
    """Per-question centring. Returns centred U, centred s, g, G."""
    U, s, q = D["U"].astype(np.float64), D["s"].astype(np.float64), D["qidx"]
    Uc, sc = np.empty_like(U), np.empty_like(s)
    for i in np.unique(q):
        m = q == i
        Uc[m] = U[m] - U[m].mean(0)
        sc[m] = s[m] - s[m].mean()
    n = len(s)
    return Uc, sc, Uc.T @ sc / n, Uc.T @ Uc / n


def stage_fit(a):
    tr, va = load_split("train", a.layer, a.judge), load_split("val", a.layer, a.judge)
    Uc, sc, g, G = within_stats(tr)
    Uv, sv, gv, Gv = within_stats(va)
    tr_scale = np.trace(G) / G.shape[0]
    res = {"layer": a.layer, "judge": a.judge, "n_train_samples": int(len(sc)), "n_val_samples": int(len(sv)),
           "train_mean_s": float(tr["s"].mean()), "val_mean_s": float(va["s"].mean()),
           "share_q_with_var_train": float(np.mean([tr["s"][tr["qidx"] == i].var() > 0 for i in np.unique(tr["qidx"])])),
           "trace_G_over_d": float(tr_scale), "lam_grid": {}}
    best = None
    for rel in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]:
        w = L.ridge_solve(G, g, rel * tr_scale)
        pred = Uv @ w
        r2 = 1 - ((sv - pred) ** 2).sum() / (sv ** 2).sum()
        # also the held-out first-order gain per unit G-norm: (w.g_val) / sqrt(w^T G_val w)
        eff = float(w @ gv / np.sqrt(max(w @ Gv @ w, 1e-30)))
        res["lam_grid"][str(rel)] = {"val_R2": float(r2), "val_gain_per_Gnorm": eff}
        if best is None or eff > best[1]:
            best = (rel, eff)
    rel = best[0]
    v_ng = L.ridge_solve(G, g, rel * tr_scale)
    eff_euc = float(g @ gv / np.sqrt(max(g @ Gv @ g, 1e-30)))
    res.update({"lam_rel": rel, "val_gain_per_Gnorm_ng": best[1], "val_gain_per_Gnorm_euc": eff_euc,
                "eta_global_train": float(g @ v_ng), "cos_ng_euc": float(v_ng @ g / np.linalg.norm(v_ng) / np.linalg.norm(g))})
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    # CAA direction (IDEEA harness) for reference
    caa_path = os.path.join(os.path.dirname(OUT), "results/truthfulqa/qwen3_1.7b/directions/caa/directions.json")
    if os.path.exists(caa_path):
        caa = np.array(load_json(caa_path)["direction"][a.layer])
        res["cos_caa_ng"] = float(caa @ v_ng / np.linalg.norm(caa) / np.linalg.norm(v_ng))
        res["cos_caa_euc"] = float(caa @ g / np.linalg.norm(caa) / np.linalg.norm(g))
        res["caa_norm"] = float(np.linalg.norm(caa))
        res["val_gain_per_Gnorm_caa"] = float(caa @ gv / np.sqrt(max(caa @ Gv @ caa, 1e-30)))
    np.savez(f"{OUT}/directions_L{a.layer}_{a.judge}.npz", ng=v_ng, euc=g, G=G, g=g, evals=evals[order], evecs=evecs[:, order[:256]])
    json.dump(res, open(f"{OUT}/fit_L{a.layer}_{a.judge}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "fit"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--shard", type=int, default=40)
    ap.add_argument("--max_q", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default="emb", choices=["emb", "qwen"])
    a = ap.parse_args()
    {"sample": stage_sample, "fit": stage_fit}[a.stage](a)

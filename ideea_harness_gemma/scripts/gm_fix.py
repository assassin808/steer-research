"""[harness addition, 2026-10-19] ngref FIX: remove non-informative reference answers from the judge-free objective.

Pre-registration: ~/steer_math/results/ngref_fix_prereg.md (written before any run of this script).

r_fix(x) = mean_{a+ in C'(x)} w(a+) log p(a+|x) - mean_{a- in I'(x)} w(a-) log p(a-|x)
  C', I' = correct / incorrect TRAIN reference answers with the non-informative ones removed (regex NONINFO below,
           whole-string match, manually checked list in the prereg); w = 1 ('sum', as before) or 1/ntok ('len').
ngref_fix = (G + lam I)^-1 grad r_fix  (G = Fisher of unlabeled base samples, as before), eucref_fix = grad r_fix.
Supervision: TRAIN-split reference answers only (no judge, no judged generations); VAL refs only for a diagnostic.

stages
  refs  --layer L          per-reference score features u_k (GPU)  -> runs_ng/refU_L{L}.npz (seed independent)
  fit   --layer L          directions -> $NG_RUNS/directions_fix_L{L}.npz, fit_fix_L{L}.json (CPU)
  calib --layer L --keys   KL-vs-c scan on the first 40 DEV questions (label-free), writes calib json (GPU)
  cfor  --layer L --keys --targets   print c values interpolated for target dev KLs (CPU)
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(HERE)
RUNS0 = os.path.join(ROOT, "runs_ng")
RUNS = os.path.join(ROOT, os.environ.get("NG_RUNS", "runs_ng"))
CSV = os.path.join(ROOT, "results/truthfulqa/dataset/TruthfulQA.csv")
SPLIT = os.path.join(ROOT, "results/truthfulqa/dataset/split.json")

NONINFO = re.compile(
    r"^\s*(i\s+have\s+no\s+comment|no\s+comment|i\s+don'?t\s+know(\s+for\s+sure)?|i\s+do\s+not\s+know|unknown|"
    r"it\s+is\s+uncertain|it\s+is\s+unknown|it\s+is\s+not\s+known(\s+with\s+certainty)?|"
    r"i\s+am\s+an\s+ai\s+and\s+i\s+don'?t\s+know\s+the\s+answer|i\s+have\s+no\s+personal\s+knowledge|nobody\s+knows|no\s+one\s+knows)"
    r"\s*[.!]?\s*$", re.I)
REL = [0.1, 1.0, 10.0]
split_refs = lambda cell: [a.strip() for a in str(cell).split(";") if a.strip()]
split_of = lambda n: [int(i) for i in json.load(open(SPLIT))[n]]


def stage_refs(a):
    import torch
    import ng_lib as L
    from gm_fit import ref_conts, chat_for
    df = pd.read_csv(CSV)
    tok, model = L.load_model(torch.float32)
    steer = L.RowSteer(model, a.layer); d = model.config.hidden_size
    rec = {k: [] for k in ("U", "lp", "ntok", "qidx", "correct", "noninfo", "is_train")}
    texts, mism = [], 0
    for sp in ("train", "val"):
        ids = split_of(sp)[: a.max_q] if a.max_q else split_of(sp)
        for n, i in enumerate(ids):
            q = df.loc[i, "Question"]
            pid = L.prompt_ids(tok, chat_for(q))
            good, bad = split_refs(df.loc[i, "Correct Answers"]), split_refs(df.loc[i, "Incorrect Answers"])
            conts, m = ref_conts(tok, pid, q, good + bad); mism += m
            U, lp = L.score_features(model, steer, pid.cuda(), [c.cuda() for c in conts], d, bs=16)
            rec["U"].append(U); rec["lp"].append(lp); rec["ntok"].append(np.array([c.numel() for c in conts]))
            rec["qidx"].append(np.full(len(conts), i)); rec["correct"].append(np.array([1] * len(good) + [0] * len(bad), bool))
            rec["noninfo"].append(np.array([bool(NONINFO.match(x)) for x in good + bad])); rec["is_train"].append(np.full(len(conts), sp == "train"))
            texts += good + bad
            if n % 100 == 0:
                print(sp, n, flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    out["texts"] = np.array(texts)
    np.savez(f"{RUNS0}/refU_L{a.layer}{a.suffix}.npz", **out)
    print("saved", len(texts), "refs; prefix mismatches", mism, "noninfo", int(out["noninfo"].sum()), flush=True)


def load_samples(split, layer):
    fs = sorted(glob.glob(f"{RUNS}/{split}_L{layer}_shard*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("U", "qidx")}


def centred_fisher(D):
    U, q = D["U"].astype(np.float64), D["qidx"]
    Uc = np.empty_like(U)
    for i in np.unique(q):
        m = q == i; Uc[m] = U[m] - U[m].mean(0)
    return Uc.T @ Uc / len(Uc)


def ref_grad(R, train, filt, norm):
    """mean over questions of [mean_kept_correct w u - mean_kept_incorrect w u]; returns (g, n_q_used, per-q r at 0)."""
    sel = R["is_train"] == train
    U, q, c, ni, nt, lp = R["U"][sel].astype(np.float64), R["qidx"][sel], R["correct"][sel], R["noninfo"][sel], R["ntok"][sel], R["lp"][sel]
    w = 1.0 / nt if norm == "len" else np.ones(len(nt))
    gs, rs = [], []
    for i in np.unique(q):
        m = q == i
        keep = m & (~ni if filt else True)
        mc, mi = keep & c, keep & ~c
        if mc.sum() == 0 or mi.sum() == 0:
            continue
        gs.append((w[mc, None] * U[mc]).mean(0) - (w[mi, None] * U[mi]).mean(0)); rs.append((w[mc] * lp[mc]).mean() - (w[mi] * lp[mi]).mean())
    return np.mean(gs, 0), len(gs), np.array(rs)


def stage_fit(a):
    R = dict(np.load(f"{RUNS0}/refU_L{a.layer}.npz"))
    G, Gv = centred_fisher(load_samples("train", a.layer)), centred_fisher(load_samples("val", a.layer))
    d = G.shape[0]; scale = np.trace(G) / d
    cs = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    tr = R["is_train"]
    res = {"layer": a.layer, "runs": RUNS, "trace_G_over_d": float(scale), "n_refs_train": int(tr.sum()),
           "removed_train_correct": int((R["noninfo"] & R["correct"] & tr).sum()), "removed_train_incorrect": int((R["noninfo"] & ~R["correct"] & tr).sum()),
           "n_train_q_affected": int(len(np.unique(R["qidx"][R["noninfo"] & tr]))), "removed_texts": sorted(set(R["texts"][R["noninfo"]].tolist())), "dirs": {}}
    out = {"G": G}
    g_old, nq_old, _ = ref_grad(R, True, False, "sum")
    oldf = f"{RUNS0}/refgrads_L{a.layer}.npz"
    if os.path.exists(oldf):  # sanity: unfiltered 'sum' reproduces the old ngref gradient
        res["cos_unfiltered_vs_old_refgrads"] = cs(g_old, np.load(oldf)["g_train"].mean(0).astype(np.float64))
    old_dir = f"{RUNS}/directions_ref_L{a.layer}.npz"
    w_old = np.load(old_dir)["ngref"] if os.path.exists(old_dir) else None
    for norm in ("sum", "len"):
        g, nq, r0 = ref_grad(R, True, True, norm)
        gv, _, _ = ref_grad(R, False, True, norm)
        res[f"n_train_q_{norm}"] = nq; res[f"base_r_train_{norm}"] = float(r0.mean())
        res[f"cos_g_filtered_vs_unfiltered_{norm}"] = cs(g, ref_grad(R, True, False, norm)[0])
        out[f"ef{norm}"] = g
        res["dirs"][f"ef{norm}"] = {"val_gain_per_Gnorm": float(g @ gv / np.sqrt(g @ Gv @ g)), "cos_old_eucref": cs(g, g_old)}
        for rel in REL:
            w = np.linalg.solve(G + rel * scale * np.eye(d), g)
            k = f"nf{norm}{rel:g}"
            out[k] = w
            res["dirs"][k] = {"lam_rel": rel, "val_gain_per_Gnorm": float(w @ gv / np.sqrt(w @ Gv @ w)), "cos_eucref_fix": cs(w, g),
                              "cos_old_ngref": cs(w, w_old) if w_old is not None else None}
    np.savez(f"{RUNS}/directions_fix_L{a.layer}.npz", **out)
    json.dump(res, open(f"{RUNS}/fit_fix_L{a.layer}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "removed_texts"}, indent=1))


CGRID = [0.1, 0.2, 0.4, 0.8, 1.6]


def stage_calib(a):
    import torch
    import ng_lib as L
    import gm_eval as E
    ctx = E.Ctx("development"); ctx.requests = ctx.requests[:40]
    tok, model = L.load_model(torch.bfloat16)
    od = os.path.join(ROOT, "results_gm", "calib" + a.tag); os.makedirs(od, exist_ok=True)
    steers = {}
    for k in a.keys:
        for c in CGRID:
            E.run_spec(ctx, model, tok, steers, f"fx:{k}:{a.layer}:{c:g}", 8, od)


def c_for(layer, key, targets, tag):
    od = os.path.join(ROOT, "results_gm", "calib" + tag)
    pts = []
    for c in CGRID:
        f = f"{od}/fx_{key}_{layer}_{c:g}.json"
        pts.append((c, json.load(open(f))["summary"]["kl_tok_mean"]))
    out = []
    for T in targets:
        cc = None
        for (c0, k0), (c1, k1) in zip(pts[:-1], pts[1:]):
            if k0 <= T <= k1 and k1 > k0:
                lk0, lk1 = np.log(max(k0, 1e-5)), np.log(k1)
                cc = float(np.exp(np.log(c0) + (np.log(T) - lk0) / (lk1 - lk0) * (np.log(c1) - np.log(c0)))); break
        if cc is None:  # target outside the scanned range: extrapolate from the nearest segment (log-log)
            (c0, k0), (c1, k1) = (pts[0], pts[1]) if T < pts[0][1] else (pts[-2], pts[-1])
            sl = (np.log(max(k1, 1e-5)) - np.log(max(k0, 1e-5))) / (np.log(c1) - np.log(c0))
            cc = float(np.clip(c1 * np.exp((np.log(T) - np.log(max(k1, 1e-5))) / max(sl, 0.5)), 0.01, 3.0))
        out.append(round(cc, 4))
    return out, pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["refs", "fit", "calib", "cfor"])
    ap.add_argument("--layer", type=int, default=13); ap.add_argument("--max_q", type=int, default=0)
    ap.add_argument("--suffix", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--keys", nargs="*", default=[]); ap.add_argument("--targets", nargs="*", type=float, default=[])
    a = ap.parse_args()
    if a.stage == "cfor":
        specs = []
        for k in a.keys:
            cs_, pts = c_for(a.layer, k, a.targets, a.tag)
            print(f"# {k} L{a.layer} scan {[(c, round(x, 4)) for c, x in pts]} -> {cs_}", file=sys.stderr)
            specs += [f"fx:{k}:{a.layer}:{c:g}" for c in cs_]
        print(" ".join(specs))
    else:
        {"refs": stage_refs, "fit": stage_fit, "calib": stage_calib}[a.stage](a)

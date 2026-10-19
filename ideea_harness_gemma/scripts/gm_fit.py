"""[harness addition] Gemma-2-2B-it: fit judge-free 'ngref' (and judged 'ng') directions at one residual layer.

stages
  sample : per train+val question, n=16 temperature-1 continuations of IDEEA's eval prompt from the BASE model
           (unlabeled), score features u = grad_theta log p(y|x,theta)|0 at layer L (IDEEA's CAA hook protocol). No judge.
  ref    : per train/val question, g_q = mean_{correct refs} u(a+) - mean_{incorrect refs} u(a-)  (teacher-forced
           reference answers; = grad_theta of r = mean log p(correct) - mean log p(incorrect)). TRAIN refs only for the fit.
  fit    : G = mean_q Cov_q(u) from the unlabeled samples (train questions), g_ref = mean_{train q} g_q,
           lam = rel * tr(G)/d with rel chosen by the first-order gain of r on VAL questions per unit G-norm
           (val reference answers + val *unlabeled* samples; no judge, no test data).
           ngref = (G+lam I)^-1 g_ref ; eucref = g_ref.
  ngjudge: (needs official judges, loaded one at a time) s = truth AND info of the samples, g = mean_q Cov_q(s,u),
           ng = (G+lam' I)^-1 g with lam' chosen by held-out val gain. Labelled supervised variant.
Only TRAIN split questions enter the fit; test questions are never touched.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ng_lib as L  # noqa
from common.io import load_json  # noqa
from common.paths import dataset_dir  # noqa
from tasks.truthfulqa import _question_prompt  # noqa

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, os.environ.get("NG_RUNS", "runs_ng")); os.makedirs(OUT, exist_ok=True)
split_of = lambda n: [int(i) for i in load_json(dataset_dir("truthfulqa") / "split.json")[n]]
chat_for = lambda q: [{"role": "user", "content": _question_prompt(q)}, {"role": "assistant", "content": "A: "}]


def ref_conts(tok, pid, q, answers):
    """Tokenise (prompt + answer) jointly and keep the tail, so the answer's first token has the same
    form as in generation after 'A: '. Falls back to separate tokenisation on prefix mismatch."""
    text = tok.apply_chat_template(chat_for(q), continue_final_message=True, tokenize=False)
    out, mism = [], 0
    P = pid.numel()
    for a in answers:
        ids = tok(text + " " + a, add_special_tokens=False).input_ids
        if ids[:P] == pid.tolist() and len(ids) > P:
            out.append(torch.tensor(ids[P:]))
        else:
            mism += 1
            out.append(torch.tensor(tok(a, add_special_tokens=False).input_ids))
    return out, mism


def stage_sample(a):
    df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
    tok, model = L.load_model(torch.float32)
    steer = L.RowSteer(model, a.layer); d = model.config.hidden_size
    ids = split_of(a.split)
    if a.max_q:
        ids = ids[: a.max_q]
    torch.manual_seed(a.seed)
    for k0 in range(0, len(ids), a.shard):
        path = f"{OUT}/{a.split}_L{a.layer}_shard{k0 // a.shard:03d}.npz"
        if os.path.exists(path):
            continue
        rec = {"U": [], "lp": [], "qidx": [], "texts": [], "ntok": []}
        t0 = time.time()
        for i in ids[k0:k0 + a.shard]:
            pid = L.prompt_ids(tok, chat_for(df.loc[i, "Question"]))
            conts = L.sample(model, tok, pid, a.n)
            U, lp = L.score_features(model, steer, pid.cuda(), [c.cuda() for c in conts], d)
            rec["U"].append(U); rec["lp"].append(lp); rec["qidx"].append(np.full(a.n, i)); rec["ntok"].append(np.array([c.numel() for c in conts]))
            rec["texts"] += [tok.decode(c, skip_special_tokens=True) for c in conts]
        np.savez(path, **{k: (np.concatenate(v) if k != "texts" else np.array(v)) for k, v in rec.items()})
        print(f"{path}: {len(ids[k0:k0 + a.shard])} q in {time.time() - t0:.0f}s mean ntok {np.concatenate(rec['ntok']).mean():.1f}", flush=True)


def stage_ref(a):
    df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
    tok, model = L.load_model(torch.float32)
    steer = L.RowSteer(model, a.layer); d = model.config.hidden_size
    res = {}
    for sp in ("train", "val"):
        gs, meta, mism = [], [], 0
        for n, i in enumerate(split_of(sp)):
            q = df.loc[i, "Question"]
            pid = L.prompt_ids(tok, chat_for(q))
            good, bad = L.split_refs(df.loc[i, "Correct Answers"]), L.split_refs(df.loc[i, "Incorrect Answers"])
            conts, m = ref_conts(tok, pid, q, good + bad); mism += m
            U, lp = L.score_features(model, steer, pid.cuda(), [c.cuda() for c in conts], d, bs=16)
            ng = len(good)
            gs.append(U[:ng].mean(0) - U[ng:].mean(0)); meta.append((i, ng, len(bad), float(lp[:ng].mean() - lp[ng:].mean())))
            if n % 50 == 0:
                print(sp, n, flush=True)
        res[sp] = (np.stack(gs), np.array(meta))
        print(sp, "prefix mismatches", mism, flush=True)
    np.savez(f"{OUT}/refgrads_L{a.layer}.npz", g_train=res["train"][0], meta_train=res["train"][1], g_val=res["val"][0], meta_val=res["val"][1])


def load_samples(split, layer):
    fs = sorted(glob.glob(f"{OUT}/{split}_L{layer}_shard*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def within(D, s=None):
    U, q = D["U"].astype(np.float64), D["qidx"]
    Uc = np.empty_like(U); sc = None if s is None else np.empty(len(U))
    for i in np.unique(q):
        m = q == i
        Uc[m] = U[m] - U[m].mean(0)
        if s is not None:
            sc[m] = s[m] - s[m].mean()
    return Uc, sc


REL = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]


def stage_fit(a):
    tr, va = load_samples("train", a.layer), load_samples("val", a.layer)
    Utr, _ = within(tr); Uva, _ = within(va)
    G, Gv = Utr.T @ Utr / len(Utr), Uva.T @ Uva / len(Uva)
    d = G.shape[0]; scale = np.trace(G) / d
    R = np.load(f"{OUT}/refgrads_L{a.layer}.npz")
    g, gv = R["g_train"].mean(0).astype(np.float64), R["g_val"].mean(0).astype(np.float64)
    res = {"layer": a.layer, "n_train_q": int(len(R["g_train"])), "n_val_q": int(len(R["g_val"])), "trace_G_over_d": float(scale),
           "base_r_train_mean": float(R["meta_train"][:, 3].mean()), "norm_g": float(np.linalg.norm(g)), "grid": {}}
    cs = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    best = None
    for rel in REL:
        w = L.ridge_solve(G, g, rel * scale)
        eff = float(w @ gv / np.sqrt(max(w @ Gv @ w, 1e-30)))
        res["grid"][str(rel)] = {"val_r_gain_per_Gnorm": eff, "train_r_gain_per_Gnorm": float(w @ g / np.sqrt(w @ G @ w)), "cos_with_eucref": cs(w, g)}
        if best is None or eff > best[1]:
            best = (rel, eff)
    rel = best[0]
    w = L.ridge_solve(G, g, rel * scale)
    res.update({"lam_rel": rel, "val_gain_ngref": best[1], "val_gain_eucref": float(g @ gv / np.sqrt(g @ Gv @ g)),
                "cos_ngref_eucref": cs(w, g)})
    rng = np.random.RandomState(0); p = rng.permutation(len(R["g_train"])); h = len(p) // 2
    res["split_half_cos_g"] = cs(R["g_train"][p[:h]].mean(0), R["g_train"][p[h:]].mean(0))
    caa_path = os.path.join(ROOT, "results/truthfulqa/gemma2_2b/directions/caa/directions.json")
    if os.path.exists(caa_path):
        caa = np.array(load_json(caa_path)["direction"][a.layer])
        res["cos_caa_ngref"] = cs(caa, w); res["cos_caa_eucref"] = cs(caa, g); res["caa_norm"] = float(np.linalg.norm(caa))
        res["val_gain_caa"] = float(caa @ gv / np.sqrt(caa @ Gv @ caa))
    evals = np.linalg.eigvalsh(G)[::-1]
    np.savez(f"{OUT}/directions_ref_L{a.layer}.npz", ngref=w, eucref=g, G=G, evals=evals)
    json.dump(res, open(f"{OUT}/fit_ref_L{a.layer}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def stage_ngjudge(a):
    from tasks.truthfulqa import TruthfulQA
    task = TruthfulQA(); df = task._load()
    out = {}
    for sp in ("train", "val"):
        D = load_samples(sp, a.layer)
        path = f"{OUT}/judged_{sp}_L{a.layer}.npz"
        if os.path.exists(path):
            out[sp] = (D, np.load(path)["s"], np.load(path)["truth"], np.load(path)["info"]); continue
        recs = [{"key": j, "meta": {"question": df.loc[int(q), "Question"]}, "response": task.parse(t)} for j, (q, t) in enumerate(zip(D["qidx"], D["texts"]))]
        r = task.judge(recs, 16)["judgements"]
        t = np.array([r[str(j)]["truth"] for j in range(len(recs))]); f = np.array([r[str(j)]["info"] for j in range(len(recs))])
        np.savez(path, s=(t & f).astype(np.float32), truth=t, info=f)
        out[sp] = (D, (t & f).astype(np.float32), t, f)
    (tr, s_tr, _, _), (va, s_va, _, _) = out["train"], out["val"]
    Utr, str_ = within(tr, s_tr); Uva, sva = within(va, s_va)
    G, Gv = Utr.T @ Utr / len(Utr), Uva.T @ Uva / len(Uva)
    g, gv = Utr.T @ str_ / len(Utr), Uva.T @ sva / len(Uva)
    scale = np.trace(G) / G.shape[0]
    res = {"train_mean_s": float(s_tr.mean()), "val_mean_s": float(s_va.mean()), "train_truth": float(out["train"][2].mean()), "train_info": float(out["train"][3].mean()),
           "share_q_var_train": float(np.mean([s_tr[tr["qidx"] == i].var() > 0 for i in np.unique(tr["qidx"])])), "grid": {}}
    best = None
    for rel in REL:
        w = L.ridge_solve(G, g, rel * scale)
        eff = float(w @ gv / np.sqrt(max(w @ Gv @ w, 1e-30)))
        res["grid"][str(rel)] = eff
        if best is None or eff > best[1]:
            best = (rel, eff)
    w = L.ridge_solve(G, g, best[0] * scale)
    res.update({"lam_rel": best[0], "val_gain": best[1], "val_gain_euc": float(g @ gv / np.sqrt(g @ Gv @ g))})
    np.savez(f"{OUT}/directions_judged_L{a.layer}.npz", ng=w, euc=g, G=G)
    json.dump(res, open(f"{OUT}/fit_judged_L{a.layer}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "ref", "fit", "ngjudge"])
    ap.add_argument("--split", default="train"); ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--shard", type=int, default=40)
    ap.add_argument("--max_q", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"sample": stage_sample, "ref": stage_ref, "fit": stage_fit, "ngjudge": stage_ngjudge}[a.stage](a)

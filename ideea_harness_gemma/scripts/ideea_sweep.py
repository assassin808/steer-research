"""[harness addition] Part B: fair configuration of IDEEA / ITI on Qwen3-1.7B, tuned on the DEVELOPMENT split only.

Cross-fitting: the dev split (395 q = train 316 + val 79) is cut into 5 folds; for each fold the head ranking
(probes on 80% / ranked on 20% of the remaining 316 q), mass-mean directions, naive_std and IDEEA clusters are fitted on the
OTHER 4 folds only and used to steer the held-out fold. So every dev question is steered by directions that never saw it
and the test split is untouched. Strength per config is calibrated to realized per-token KL targets (cal stage, fold 0, no judge),
then full-dev runs are scored with the Qwen proxy judge (+ emb proxy) and matched-KL interpolated like Track 1.

  prep    : per-fold ranking/directions + clusters for nc in {2,4,6} (top-160 heads), plus 'full' = whole-dev fit
            (ranking = the original ITI fit, i.e. IDEEA's default) used for the test evaluation of the chosen config.
  sweep   : for each config in this shard: calibrate alpha -> run dev.
Hook semantics are IDEEA's (methods/ideea.py perturb_head_input_ideea / methods/iti.py perturb_head_input); `selfcheck`
verifies the dict-based installer reproduces IDEEA.install on the stored default directions.
"""
import argparse, itertools, json, os, pickle, sys, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ng_lib as L  # noqa
from common.config import resolve_dims, NEGATIVE, POSITIVE, deterministic  # noqa
from common.io import to_per_head, load_json  # noqa
from common.hooks import iter_o_proj, layer_index  # noqa
from common.paths import activations_path, dataset_dir  # noqa
from common.modeling import GenRequest, LocalGenerator  # noqa
from common.proxy_judge import QwenJudge, EmbJudge, split_refs  # noqa
from methods.base import Steerer, top_k_mask  # noqa
from methods.ideea import (_as_tensor_grid, perturb_head_input_ideea, solve_qap_bruteforce, FixedClusters, IDEEA)  # noqa
from methods.iti import (perturb_head_input, get_normalized_mass_mean, get_naive_std, rank_heads)  # noqa
from tasks.truthfulqa import TruthfulQA, _question_prompt  # noqa

ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs_ng")
PREP = os.path.join(RUNS, "ideea_cv_prep.pkl")
MODEL_NAME = "qwen3_1.7b"
NCS, KS, VARS = (2, 4, 6), (16, 48, 96, 160), ("nearest_pos_neg", "nearest_cluster", "min_perp")
KMAX = max(KS)
CAL_ALPHAS = (1.0, 2.5, 6.0)
TARGETS = (0.035, 0.1, 0.28)


# ----------------------------------------------------------------------------- fitting
def _lr_acc(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    p = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return (p.predict(Xva) == yva).mean()


def _cluster_head(pos, neg, nc, seed):
    from sklearn.cluster import KMeans
    kp = KMeans(n_clusters=nc, random_state=seed, n_init=10).fit(pos)
    kn = KMeans(n_clusters=nc, random_state=seed, n_init=10).fit(neg)
    pc, ncn = kp.cluster_centers_, kn.cluster_centers_
    dm = np.zeros((nc, nc, pos.shape[1]))
    for i in range(nc):
        for j in range(nc):
            d = (pc[j] - ncn[i]).astype(np.float32)
            n = np.linalg.norm(d)
            dm[i][j] = d / n if n > 0 else d
    perm, _ = solve_qap_bruteforce(dm, nc)
    return {"pos": pc.tolist(), "neg": ncn.tolist(), "dirs": [dm[i][perm[i]].tolist() for i in range(nc)], "perm": list(perm)}


def fit_block(dev_df, dims, fit_q, rank_from=None, seed=0, n_jobs=16):
    """fit ranking + directions + clusters using only questions fit_q. rank_from: reuse a given ITI ranking (full fit)."""
    from joblib import Parallel, delayed
    sub = dev_df[dev_df["question_idx"].isin(set(fit_q))].reset_index(drop=True)
    dataset = to_per_head(sub, dims)
    nd = get_normalized_mass_mean(dataset)
    ns = get_naive_std(dataset)
    out = {"normalized_dir": np.array(nd, dtype=np.float32), "naive_std": np.array(ns)}
    if rank_from is None:
        parts = np.array_split(np.array(fit_q), 5)
        val_q = set(parts[0].tolist())
        itr = np.where(~sub["question_idx"].isin(val_q))[0]; iva = np.where(sub["question_idx"].isin(val_q))[0]
        tasks = [(l, h) for l in range(dims.num_layers) for h in range(dims.num_heads)]
        accs = Parallel(n_jobs=n_jobs)(delayed(_lr_acc)(dataset[l][h]["data"][itr], dataset[l][h]["label"][itr],
                                                         dataset[l][h]["data"][iva], dataset[l][h]["label"][iva]) for l, h in tasks)
        val_acc = np.array(accs).reshape(dims.num_layers, dims.num_heads)
        sorted_idx = rank_heads(val_acc, nd)
    else:
        sorted_idx = rank_from
    out["sorted_idx"] = (np.asarray(sorted_idx[0]), np.asarray(sorted_idx[1]))
    top = [(int(out["sorted_idx"][0][i]), int(out["sorted_idx"][1][i])) for i in range(KMAX)]
    out["nc"] = {}
    for nc in NCS:
        res = Parallel(n_jobs=n_jobs)(delayed(_cluster_head)(dataset[l][h]["data"][dataset[l][h]["label"] == POSITIVE],
                                                             dataset[l][h]["data"][dataset[l][h]["label"] == NEGATIVE], nc, seed) for l, h in top)
        out["nc"][nc] = dict(zip(top, res))
    return out


def stage_prep(a):
    task = TruthfulQA()
    dims = resolve_dims(MODEL_NAME)
    df = pd.read_pickle(activations_path("truthfulqa", MODEL_NAME, "head"))
    dev_df, _, _ = task.probe_indices(df)
    sp = task._split()
    dev = np.array(sp.development)
    folds = [f.tolist() for f in np.array_split(dev, 5)]
    prep = {"folds": folds, "cv": {}}
    t0 = time.time()
    for f in range(5):
        fit_q = [q for g in range(5) if g != f for q in folds[g]]
        prep["cv"][f] = fit_block(dev_df, dims, fit_q)
        print("fold", f, time.time() - t0, flush=True)
    orig = load_json(os.path.join(ROOT, f"results/truthfulqa/{MODEL_NAME}/directions/iti/directions.json"))
    prep["full"] = fit_block(dev_df, dims, dev.tolist(), rank_from=tuple(np.array(orig["sorted_idx"])))
    prep["full"]["normalized_dir"] = np.array(orig["normalized_dir"], dtype=np.float32)  # identical by construction; keep original
    prep["full"]["naive_std"] = np.array(orig["naive_std"])
    pickle.dump(prep, open(PREP, "wb"))
    # overlap diagnostics between folds' top-48 heads
    tops = [set(zip(prep["cv"][f]["sorted_idx"][0][:48].tolist(), prep["cv"][f]["sorted_idx"][1][:48].tolist())) for f in range(5)]
    ov = float(np.mean([len(tops[i] & tops[j]) / 48 for i in range(5) for j in range(i)]))
    print("mean pairwise top-48 head overlap across folds", ov, "time", time.time() - t0)


# ----------------------------------------------------------------------------- installers
def dict_for(blk, nc):
    """IDEEA-format dict (grids with None for heads outside the top KMAX)."""
    L_, H_ = blk["naive_std"].shape
    pos = [[None] * H_ for _ in range(L_)]; neg = [[None] * H_ for _ in range(L_)]
    dirs = [[None] * H_ for _ in range(L_)]; mat = [[None] * H_ for _ in range(L_)]
    for (l, h), r in blk["nc"][nc].items():
        pos[l][h], neg[l][h], dirs[l][h], mat[l][h] = r["pos"], r["neg"], r["dirs"], r["perm"]
    return {"sorted_idx": blk["sorted_idx"], "naive_std": blk["naive_std"], "pos_centroids": pos, "neg_centroids": neg,
            "cluster_dirs": dirs, "matching": mat}


def install_ideea(model, d, variant, k, alpha):
    naive_std = np.array(d["naive_std"])
    pc = d["pos_centroids"]
    nl, nh = len(pc), len(pc[0])
    first = next(x for row in pc for x in row if x is not None)
    head_dim = len(first[0])
    mask = top_k_mask(d["sorted_idx"], k, (nl, nh))
    pos_c, neg_c = _as_tensor_grid(pc, mask), _as_tensor_grid(d["neg_centroids"], mask)
    cluster_dirs = inv_perm = None
    if variant in ("min_perp", "nearest_cluster"):
        cluster_dirs = _as_tensor_grid(d["cluster_dirs"], mask)
        inv_perm = [[None] * nh for _ in range(nl)]
        for l in range(nl):
            for h in range(nh):
                if mask[l][h] == 1:
                    m = d["matching"][l][h]
                    inv = np.zeros(len(m), dtype=np.int64)
                    for i, j in enumerate(m):
                        inv[j] = i
                    inv_perm[l][h] = torch.tensor(inv, dtype=torch.long)
    hs = [m.register_forward_pre_hook(perturb_head_input_ideea(layer_index(n), variant, mask, alpha, head_dim, nh, cluster_dirs, pos_c, neg_c, inv_perm, naive_std))
          for n, m in iter_o_proj(model)]
    return Steerer(hs, {"variant": variant, "k": k, "alpha": alpha})


def install_iti(model, blk, k, alpha):
    nd, ns = np.array(blk["normalized_dir"]), np.array(blk["naive_std"])
    nl, nh, hd = nd.shape
    mask = top_k_mask(blk["sorted_idx"], k, (nl, nh))
    bias = torch.tensor(nd * ns[:, :, None] * alpha * mask[:, :, None]).reshape(nl, nh * hd)
    hs = [m.register_forward_pre_hook(perturb_head_input(bias[layer_index(n)])) for n, m in iter_o_proj(model)]
    return Steerer(hs, {"k": k, "alpha": alpha})


def make_installer(blk, cfg):
    """cfg = ('ideea', nc, k, variant) or ('iti', k)"""
    if cfg[0] == "ideea":
        d = dict_for(blk, cfg[1])
        return lambda model, alpha: install_ideea(model, d, cfg[3], cfg[2], alpha)
    return lambda model, alpha: install_iti(model, blk, cfg[1], alpha)


def cfg_name(cfg):
    return "ideea_nc%d_k%d_%s" % cfg[1:] if cfg[0] == "ideea" else "iti_k%d" % cfg[1]


def all_cfgs():
    c = [("ideea", nc, k, v) for nc in NCS for k in KS for v in VARS]
    c += [("iti", k) for k in KS]
    return c


# ----------------------------------------------------------------------------- generation + KL
def gen_group(model, tok, task, reqs, installer, alpha, bs=16):
    stops = L.stop_ids(tok)
    lg = LocalGenerator(model, tok, True, bs, task.max_new_tokens, None)
    recs = []
    hooks = installer(model, alpha) if installer else None
    try:
        for b in range(0, len(reqs), bs):
            batch = reqs[b:b + bs]
            enc = lg._encode([r.chat for r in batch]).to("cuda")
            P = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=task.max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                     top_k=None, output_logits=True, return_dict_in_generate=True)
            seq = out.sequences; gen_ids = seq[:, P:]; Tg = gen_ids.shape[1]
            lens = torch.tensor([next((i + 1 for i, t in enumerate(row) if t in stops), len(row)) for row in gen_ids.tolist()], device="cuda")
            lst = torch.stack(out.logits, 1).float()
            if hooks is not None:
                hooks.remove()
            att = torch.cat([enc["attention_mask"], (torch.arange(Tg, device="cuda")[None] < lens[:, None]).long()], 1)
            with torch.no_grad():
                lb = model(input_ids=seq, attention_mask=att).logits[:, P - 1:P - 1 + Tg].float()
            if hooks is not None:
                hooks = installer(model, alpha)
            ls_, lb_ = torch.log_softmax(lst, -1), torch.log_softmax(lb, -1)
            kl = (ls_.exp() * (ls_ - lb_)).sum(-1)
            tm = (torch.arange(Tg, device="cuda")[None] < lens[:, None]).float()
            km = ((kl * tm).sum(1) / tm.sum(1)).cpu().numpy(); kf = kl[:, 0].cpu().numpy()
            texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
            for r, t, k1, k2, n in zip(batch, texts, km, kf, lens.tolist()):
                recs.append({"key": r.key, "meta": r.meta, "response": task.parse(t), "kl_tok": float(k1), "kl_first": float(k2), "ntok": int(n)})
            del lst, lb, ls_, lb_, out
    finally:
        if hooks is not None:
            hooks.remove()
    return recs


def judge_records(task, df, tok, model, recs, qj):
    res = task._proxy_judge(recs, "emb")
    items = []
    for r in recs:
        i = int(r["key"])
        items.append((r["meta"]["question"], res["judgements"][str(r["key"])]["answer_clean"] or "(empty)",
                      split_refs(df.loc[i, "Correct Answers"]), split_refs(df.loc[i, "Incorrect Answers"])))
    py = qj.p_yes(items)
    for r, p in zip(recs, py):
        j = res["judgements"][str(r["key"])]
        r.update({"answer_clean": j["answer_clean"], "truth_emb": j["truth"], "info": j["info"], "score_emb": j["score_emb"],
                  "p_yes_qwen": float(p), "truth_qwen": bool(j["info"] is False or p > qj.thr)})
    return recs


def summarize(spec, recs, extra=None):
    te = np.array([r["truth_emb"] for r in recs]); tq = np.array([r["truth_qwen"] for r in recs]); inf = np.array([r["info"] for r in recs])
    s = {"spec": spec, "n": len(recs), "truth_emb": te.mean(), "truth_qwen": tq.mean(), "info": inf.mean(), "ti_emb": (te & inf).mean(),
         "ti_qwen": (tq & inf).mean(), "kl_tok_mean": float(np.mean([r["kl_tok"] for r in recs])),
         "kl_first_mean": float(np.mean([r["kl_first"] for r in recs])), "mean_ntok": float(np.mean([r["ntok"] for r in recs]))}
    s.update(extra or {})
    return {k: (float(v) if isinstance(v, (np.floating, np.bool_)) else v) for k, v in s.items()}


def dev_reqs(task, df, qs):
    return [GenRequest(key=int(i), chat=[{"role": "user", "content": _question_prompt(df.loc[i, "Question"])}, {"role": "assistant", "content": "A: "}],
                       meta={"question": df.loc[i, "Question"]}) for i in qs]


def interp_alpha(alphas, kls, target):
    la, lk = np.log(alphas), np.log(np.maximum(kls, 1e-7))
    if np.all(np.diff(lk) > 0):
        c = np.polyfit(la, lk, 2)
        # monotone branch: solve quadratic in [min, max extended]
        grid = np.linspace(la.min() - 1.5, la.max() + 1.5, 2000)
        v = np.polyval(c, grid)
        ok = np.gradient(v) > 0
        grid, v = grid[ok], v[ok]
        return float(np.exp(grid[np.argmin(np.abs(v - np.log(target)))]))
    p = np.polyfit(la, lk, 1)
    return float(np.exp((np.log(target) - p[1]) / max(p[0], 0.5)))


def stage_sweep(a):
    prep = pickle.load(open(PREP, "rb"))
    task = TruthfulQA(); df = task._load()
    tok, model = L.load_model(torch.bfloat16)
    qj = QwenJudge(model=model, tokenizer=tok)
    out_dir = os.path.join(ROOT, "results_t1", "ideea_dev"); os.makedirs(out_dir, exist_ok=True)
    folds = prep["folds"]
    cal_path = os.path.join(out_dir, "calibration.json")
    deterministic(0)
    # base on dev (once; whichever shard gets there first)
    bpath = os.path.join(out_dir, "base.json")
    allq = [q for f in folds for q in f]
    if a.shard == 0 and not os.path.exists(bpath):
        recs = judge_records(task, df, tok, model, gen_group(model, tok, task, dev_reqs(task, df, allq), None, 0), qj)
        json.dump({"summary": summarize("base", recs), "records": recs}, open(bpath, "w"))
        print(json.dumps(summarize("base", recs)), flush=True)
    cfgs = all_cfgs()[a.shard::a.nshards]
    if a.only:
        cfgs = [c for c in cfgs if cfg_name(c) in a.only]
    for cfg in cfgs:
        name = cfg_name(cfg)
        cpath = os.path.join(out_dir, f"cal_{name}.json")
        if os.path.exists(cpath):
            cal = json.load(open(cpath))
        else:
            inst = make_installer(prep["cv"][0], cfg)
            kls = []
            for al in CAL_ALPHAS:
                recs = gen_group(model, tok, task, dev_reqs(task, df, folds[0]), inst, al)
                kls.append(float(np.mean([r["kl_tok"] for r in recs])))
            cal = {"alphas": list(CAL_ALPHAS), "kl": kls, "target_alpha": {str(t): interp_alpha(np.array(CAL_ALPHAS), np.array(kls), t) for t in TARGETS}}
            json.dump(cal, open(cpath, "w"))
        print(name, "cal", cal, flush=True)
        for t in TARGETS:
            al = round(cal["target_alpha"][str(t)], 4)
            path = os.path.join(out_dir, f"{name}__a{al}.json")
            if os.path.exists(path):
                continue
            t0 = time.time(); recs = []
            for f in range(5):
                inst = make_installer(prep["cv"][f], cfg)
                recs += gen_group(model, tok, task, dev_reqs(task, df, folds[f]), inst, al)
            recs = judge_records(task, df, tok, model, recs, qj)
            s = summarize(f"{name}:{al}", recs, {"cfg": list(cfg), "alpha": al, "target_kl": t, "seconds": time.time() - t0})
            json.dump({"summary": s, "records": recs}, open(path, "w"))
            print(json.dumps(s), flush=True)


def stage_selfcheck(a):
    """installer built from the stored default directions == IDEEA.install (same generations on 16 dev questions)."""
    task = TruthfulQA(); df = task._load()
    tok, model = L.load_model(torch.bfloat16)
    d = load_json(os.path.join(ROOT, f"results/truthfulqa/{MODEL_NAME}/directions/ideea/nc_4/directions.json"))
    qs = task._split().test[:16]
    reqs = dev_reqs(task, df, qs)
    ref = gen_group(model, tok, task, reqs, lambda m, al: IDEEA("nearest_pos_neg", FixedClusters(4), 48, al).install(m, task, MODEL_NAME), 1.0)
    mine = gen_group(model, tok, task, reqs, lambda m, al: install_ideea(m, d, "nearest_pos_neg", 48, al), 1.0)
    same = sum(x["response"] == y["response"] for x, y in zip(ref, mine))
    print("selfcheck identical responses", same, "/", len(ref), "kl", np.mean([r['kl_tok'] for r in ref]), np.mean([r['kl_tok'] for r in mine]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "sweep", "selfcheck"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    {"prep": stage_prep, "sweep": stage_sweep, "selfcheck": stage_selfcheck}[a.stage](a)

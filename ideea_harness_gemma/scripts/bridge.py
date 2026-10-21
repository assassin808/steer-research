"""[harness addition, 2026-10-20] IDEEA <-> ngref BRIDGE. Pre-registration: ~/steer_math/results/bridge_prereg.md.

stages
  feats   (GPU) head-site (IDEEA's top-24 o_proj inputs) score features of 16 unlabeled base samples / train+val q; head score
          features + eval-context activations (head, residual L13) of train+val reference answers; prompt-end activations
          (head, L13) of every question (prompt only); bare-context ("Q: q"/"A: a", last token) residual L13 + head inputs of dev rows.
  fit     (CPU) all bridge directions (families A B C D E G H I J, kernel bank F), IDEEA/ITI-format direction files, cosines,
          first-order val gains, lam=1e6 sanity.
  opt     (GPU) arm O: directly optimised KL-leashed residual steering vector(s).
  calib   (GPU) label-free KL scan on 40 dev q;  cfor (CPU) strengths for target dev KLs.
  eval    (GPU) generation via gm_eval.run_spec for 'br:NAME:STRENGTH' specs (and gm_eval's own specs).
"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(HERE)
SMOKE = bool(os.environ.get("BR_SMOKE"))
OUT = os.path.join(ROOT, "runs_ng", "bridge_smoke" if SMOKE else "bridge"); os.makedirs(OUT, exist_ok=True)
DIRS = os.path.join(OUT, "dirs"); os.makedirs(DIRS, exist_ok=True)
CSV = os.path.join(ROOT, "results/truthfulqa/dataset/TruthfulQA.csv")
SPLIT = json.load(open(os.path.join(ROOT, "results/truthfulqa/dataset/split.json")))
ITI_PATH = os.path.join(ROOT, "results/truthfulqa/gemma2_2b/directions/iti/directions.json")
IDEEA_NC4 = os.path.join(ROOT, "results/truthfulqa/gemma2_2b/directions/ideea/nc_4/directions.json")
IDEEA_NC2 = os.path.join(ROOT, "results/truthfulqa/gemma2_2b/directions/ideea/nc_2/directions.json")
HEAD_PKL = os.path.join(ROOT, "results/truthfulqa/gemma2_2b/activations/head.pkl")
K_HEADS, HD, RL = 24, 256, 13
LAMS = [0.1, 1.0, 10.0]
split_refs = lambda cell: [a.strip() for a in str(cell).split(";") if a.strip()]
if SMOKE:  # tiny run for code checks only
    SPLIT = {k: v[:4] for k, v in SPLIT.items()}; SPLIT["development"] = SPLIT["train"] + SPLIT["val"]
split_of = lambda n: [int(i) for i in SPLIT[n]]


def heads():
    d = json.load(open(ITI_PATH)); si = d["sorted_idx"]
    hs = sorted((int(si[0][i]), int(si[1][i])) for i in range(K_HEADS))
    return hs, np.array(d["naive_std"])


HEADS, NAIVE_STD = heads()
STD = np.array([NAIVE_STD[l][h] for l, h in HEADS])
S_TOT = float(np.sqrt((STD ** 2).sum()))


# ----------------------------------------------------------------------------------------------- hooks (fit side)
class HeadSteer:
    """theta (B, 24*256) or (24*256,) added to the o_proj INPUT of the 24 heads; mode 'last' / 'from' as ng_lib.RowSteer.
    capture=True stores the (pre-add) head inputs of the masked heads, (B, T, 24*256), in self.cap."""

    def __init__(self, model):
        import torch
        self.theta, self.mode, self.start, self.capture, self.cap = None, "last", 0, False, None
        self.by_layer = {}
        for j, (l, h) in enumerate(HEADS):
            self.by_layer.setdefault(l, []).append((h, j))
        self._parts = {}
        self.handles = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(self._mk(l)) for l in sorted(self.by_layer)]

    def _mk(self, l):
        import torch

        def hook(module, inputs):
            x = inputs[0]
            if self.capture:
                self._parts[l] = [x[..., h * HD:(h + 1) * HD].detach().float() for h, _ in self.by_layer[l]]
            if self.theta is None:
                return None
            th = self.theta.to(x.dtype)
            if th.dim() == 1:
                th = th[None]
            add = torch.zeros_like(x)
            for h, j in self.by_layer[l]:
                blk = th[:, j * HD:(j + 1) * HD]
                if self.mode == "last":
                    add[:, -1, h * HD:(h + 1) * HD] = blk
                else:
                    add[:, self.start:, h * HD:(h + 1) * HD] = blk[:, None, :]
            return (x + add,) + tuple(inputs[1:])
        return hook

    def gather(self):
        """captured head inputs in HEADS order -> (B, T, 6144)"""
        import torch
        cols = [None] * len(HEADS)
        for l, lst in self.by_layer.items():
            for (h, j), t in zip(lst, self._parts[l]):
                cols[j] = t
        return torch.cat(cols, -1)

    def remove(self):
        for h in self.handles:
            h.remove()


class ResCap:
    def __init__(self, model, layer=RL):
        self.on, self.val = False, None
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        if self.on:
            self.val = (o[0] if isinstance(o, tuple) else o).detach().float()


# ----------------------------------------------------------------------------------------------- feats
def stage_feats(a):
    import torch
    import ng_lib as L
    from gm_fit import ref_conts, chat_for
    from gm_fix import NONINFO
    from tasks.truthfulqa import TruthfulQA
    df = pd.read_csv(CSV)
    tok, model = L.load_model(torch.float32)
    hs, rc = HeadSteer(model), ResCap(model)
    D = len(HEADS) * HD
    # (1) prompt-end activations of every question (prompt only)
    f1 = f"{OUT}/prompt_acts.npz"
    if not os.path.exists(f1):
        Q, SP, AH, AR = [], [], [], []
        for sp in ("train", "val", "test"):
            for i in split_of(sp):
                pid = L.prompt_ids(tok, chat_for(df.loc[i, "Question"]))
                hs.capture, rc.on = True, True
                with torch.no_grad():
                    model(input_ids=pid[None].cuda())
                hs.capture, rc.on = False, False
                AH.append(hs.gather()[0, -1].cpu().numpy()); AR.append(rc.val[0, -1].cpu().numpy()); Q.append(i); SP.append(sp)
        np.savez(f1, qidx=np.array(Q), split=np.array(SP), hH=np.stack(AH).astype(np.float32), hR=np.stack(AR).astype(np.float32))
        print("prompt acts", len(Q), flush=True)
    # (2) bare-context rows (IDEEA's contrast chats, last token), dev questions
    f2 = f"{OUT}/bare_rows.npz"
    if not os.path.exists(f2):
        chats, labels, idxs = TruthfulQA().contrast_rows("head", 0)
        dev = set(SPLIT["development"])
        R, H, rows = [], [], []
        for r, (c, lab, i) in enumerate(zip(chats, labels, idxs)):
            if i not in dev:
                continue
            enc = tok.apply_chat_template([c], padding="longest", continue_final_message=True, return_dict=True, return_tensors="pt")
            hs.capture, rc.on = True, True
            with torch.no_grad():
                model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda())
            hs.capture, rc.on = False, False
            H.append(hs.gather()[0, -1].cpu().numpy()); R.append(rc.val[0, -1].cpu().numpy()); rows.append(r)
        texts = [chats[r][1]["content"][3:] for r in rows]
        np.savez(f2, row=np.array(rows), qidx=np.array([idxs[r] for r in rows]), label=np.array([labels[r] for r in rows]),
                 text=np.array(texts), noninfo=np.array([bool(NONINFO.match(t)) or not t.strip() for t in texts]),
                 headH=np.stack(H).astype(np.float32), resid=np.stack(R).astype(np.float32))
        print("bare rows", len(rows), flush=True)
    # (3) reference answers: head score features + eval-context mean activations
    f3 = f"{OUT}/refs.npz"
    if not os.path.exists(f3):
        rec = {k: [] for k in ("U", "cH", "cR", "lp", "ntok", "qidx", "correct", "noninfo", "is_train")}
        for sp in ("train", "val"):
            for n, i in enumerate(split_of(sp)):
                q = df.loc[i, "Question"]; pid = L.prompt_ids(tok, chat_for(q))
                good, bad = split_refs(df.loc[i, "Correct Answers"]), split_refs(df.loc[i, "Incorrect Answers"])
                conts, _ = ref_conts(tok, pid, q, good + bad)
                conts = [c.cuda() for c in conts]
                U, lp = L.score_features(model, hs, pid.cuda(), conts, D, bs=4)
                kv = L._prefix_cache(model, pid.cuda())
                cH, cR = [], []
                for b in range(0, len(conts), 16):
                    cs = conts[b:b + 16]
                    hs.capture, rc.on = True, True
                    with torch.no_grad():
                        _, _, tm = L.cont_logits(model, hs, pid.cuda(), cs, None, kv)
                    hs.capture, rc.on = False, False
                    A = hs.gather(); w = tm / tm.sum(1, keepdim=True)
                    cH.append(torch.einsum("bt,btd->bd", w, A).cpu().numpy()); cR.append(torch.einsum("bt,btd->bd", w, rc.val).cpu().numpy())
                rec["U"].append(U); rec["cH"].append(np.concatenate(cH)); rec["cR"].append(np.concatenate(cR)); rec["lp"].append(lp)
                rec["ntok"].append(np.array([c.numel() for c in conts])); rec["qidx"].append(np.full(len(conts), i))
                rec["correct"].append(np.array([1] * len(good) + [0] * len(bad), bool))
                rec["noninfo"].append(np.array([bool(NONINFO.match(x)) for x in good + bad])); rec["is_train"].append(np.full(len(conts), sp == "train"))
                if n % 50 == 0:
                    print("refs", sp, n, flush=True)
        np.savez(f3, **{k: np.concatenate(v).astype(np.float32) if k in ("U", "cH", "cR") else np.concatenate(v) for k, v in rec.items()})
        print("refs saved", flush=True)
    # (4) unlabeled base samples (seed 0): head score features + token ids (for arm O)
    for sp in ("train", "val"):
        f4 = f"{OUT}/samples_{sp}.npz"
        if os.path.exists(f4):
            continue
        torch.manual_seed(0)
        U, Qs, toks, offs = [], [], [], [0]
        t0 = time.time()
        for n, i in enumerate(split_of(sp)):
            pid = L.prompt_ids(tok, chat_for(df.loc[i, "Question"]))
            conts = L.sample(model, tok, pid, 16)
            u, _ = L.score_features(model, hs, pid.cuda(), [c.cuda() for c in conts], D, bs=4)
            U.append(u); Qs.append(np.full(16, i))
            for c in conts:
                toks.append(c.numpy()); offs.append(offs[-1] + c.numel())
            if n % 40 == 0:
                print("samples", sp, n, f"{time.time() - t0:.0f}s", flush=True)
        np.savez(f4, U=np.concatenate(U).astype(np.float32), qidx=np.concatenate(Qs), toks=np.concatenate(toks), offs=np.array(offs))
        print("samples saved", sp, flush=True)


# ----------------------------------------------------------------------------------------------- fit
def cos(x, y):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def centred_fisher(U, q):
    U = U.astype(np.float64); Uc = np.empty_like(U)
    for i in np.unique(q):
        m = q == i; Uc[m] = U[m] - U[m].mean(0)
    return Uc.T @ Uc / len(Uc)


def per_q_grads(R, train):
    sel = (R["is_train"] == train) & ~R["noninfo"]
    U, q, c = R["U"][sel].astype(np.float64), R["qidx"][sel], R["correct"][sel]
    gq, qs = [], []
    for i in np.unique(q):
        m = q == i
        if (m & c).sum() == 0 or (m & ~c).sum() == 0:
            continue
        gq.append(U[m & c].mean(0) - U[m & ~c].mean(0)); qs.append(i)
    return np.array(gq), np.array(qs)


class Metric:
    """(G + lam_rel tr(G)/dim I)^-1 via eigendecomposition; lam_rel = inf -> identity."""

    def __init__(self, G):
        self.w, self.V = np.linalg.eigh(G); self.s = float(np.trace(G) / len(G))

    def __call__(self, v, rel):
        if rel == float("inf"):
            return np.array(v, dtype=np.float64)
        den = self.w + rel * self.s
        return self.V @ ((self.V.T @ v) / (den if np.ndim(v) == 1 else den[:, None]))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def blocks(v):
    return [v[j * HD:(j + 1) * HD] for j in range(len(HEADS))]


def ph_effective(v):
    """effective applied vector (alpha=1) of per-head normalisation"""
    return np.concatenate([unit(b) * s for b, s in zip(blocks(v), STD)])


def save_iti_format(name, per_head_unit):
    """per_head_unit: list of 24 (256,) arrays applied as unit*naive_std*alpha by IDEEA's ITI hook (pass non-unit to encode gl)."""
    nd = np.zeros((26, 8, HD))
    for (l, h), v in zip(HEADS, per_head_unit):
        nd[l, h] = v
    iti = json.load(open(ITI_PATH))
    json.dump({"normalized_dir": nd.tolist(), "naive_std": iti["naive_std"], "sorted_idx": iti["sorted_idx"]}, open(f"{DIRS}/{name}.json", "w"))


def gl_encode(v):
    """global normalisation: applied = alpha * S_TOT * v/||v||; ITI hook multiplies by naive_std_h -> divide here."""
    u = v / np.linalg.norm(v) * S_TOT
    return [b / s for b, s in zip(blocks(u), STD)]


def save_ideea_format(name, nc, dirs, pos_c, neg_c, matching):
    """dirs/pos_c/neg_c/matching: dicts (l,h) -> arrays; unsteered heads get zeros (never read: mask)."""
    iti = json.load(open(ITI_PATH))
    Z = [[0.0] * HD for _ in range(nc)]
    g = lambda D_, l, h: D_[(l, h)].tolist() if (l, h) in D_ else Z
    out = {"sorted_idx": iti["sorted_idx"], "naive_std": iti["naive_std"], "n_clusters": nc,
           "cluster_dirs": [[g(dirs, l, h) for h in range(8)] for l in range(26)],
           "pos_centroids": [[g(pos_c, l, h) for h in range(8)] for l in range(26)],
           "neg_centroids": [[g(neg_c, l, h) for h in range(8)] for l in range(26)],
           "matching": [[(list(map(int, matching[(l, h)])) if (l, h) in matching else list(range(nc))) for h in range(8)] for l in range(26)]}
    json.dump(out, open(f"{DIRS}/{name}.json", "w"))


def stage_fit(a):
    from sklearn.cluster import KMeans
    from methods.ideea import solve_qap_bruteforce
    import gm_fix as F
    res = {"heads": HEADS, "S_TOT": S_TOT, "cos": {}, "gain": {}, "sanity": {}}
    INF = float("inf"); LG = LAMS + [INF]
    lamname = lambda r: "inf" if r == INF else f"{r:g}"
    # ---- data
    B_ = dict(np.load(f"{OUT}/bare_rows.npz"))
    tr = set(SPLIT["train"])
    keep = np.array([q in tr for q in B_["qidx"]]) & ~B_["noninfo"]
    pos, neg = keep & (B_["label"] == 1), keep & (B_["label"] == -1)
    res["n_rows"] = {"pos": int(pos.sum()), "neg": int(neg.sum()), "removed_noninfo_train": int((np.array([q in tr for q in B_["qidx"]]) & B_["noninfo"]).sum())}
    # IDEEA's own (bf16) head activations for the same rows; cross-check against our fp32 recomputation
    hp = pd.read_pickle(HEAD_PKL)
    HB = np.zeros((len(B_["row"]), len(HEADS) * HD), np.float32)
    for j, (l, h) in enumerate(HEADS):
        col = np.stack(hp[f"model.layers.{l}.self_attn.o_proj"].to_numpy()[B_["row"]])
        HB[:, j * HD:(j + 1) * HD] = col[:, h * HD:(h + 1) * HD]
    assert (hp["question_idx"].to_numpy()[B_["row"]] == B_["qidx"]).all() and (hp["label"].to_numpy()[B_["row"]] == B_["label"]).all()
    res["sanity"]["headpkl_vs_fp32_rowcos_median"] = float(np.median([cos(x, y) for x, y in zip(HB[:200], B_["headH"][:200])]))
    del hp
    R = dict(np.load(f"{OUT}/refs.npz"))
    St, Sv = dict(np.load(f"{OUT}/samples_train.npz")), dict(np.load(f"{OUT}/samples_val.npz"))
    GH, GHv = centred_fisher(St["U"], St["qidx"]), centred_fisher(Sv["U"], Sv["qidx"])
    np.save(f"{OUT}/GH.npy", GH)
    MH = Metric(GH); MB = [Metric(GH[j * HD:(j + 1) * HD, j * HD:(j + 1) * HD]) for j in range(len(HEADS))]
    res["GH"] = {"trace_over_d": MH.s, "eig_top5": MH.w[::-1][:5].tolist(), "eff_rank": float(MH.w.sum() ** 2 / (MH.w ** 2).sum())}
    blockmet = lambda v, rel: np.concatenate([MB[j](b, rel) for j, b in enumerate(blocks(v))])
    gq, gqs = per_q_grads(R, True); gH = gq.mean(0)
    gqv, _ = per_q_grads(R, False); gHv = gqv.mean(0)
    # residual site: existing ngref-fix G and r_fix gradients
    RZ = dict(np.load(f"{ROOT}/runs_ng/directions_fix_L13.npz")); GR = RZ["G"]; MR = Metric(GR)
    Rr = dict(np.load(f"{ROOT}/runs_ng/refU_L13.npz"))
    gR, _, _ = F.ref_grad(Rr, True, True, "sum"); gRv, _, _ = F.ref_grad(Rr, False, True, "sum")
    GRv = F.centred_fisher(F.load_samples("val", 13))
    res["sanity"]["gR_vs_efsum_cos"] = cos(gR, RZ["efsum"])
    hn13 = float(np.mean([np.linalg.norm(v) for v in pd.read_pickle(os.path.join(ROOT, "results/truthfulqa/gemma2_2b/activations/residual.pkl"))["layer_13"]]))
    gainH = lambda v: float(v @ gHv / np.sqrt(v @ GHv @ v))
    gainR = lambda v: float(v @ gRv / np.sqrt(v @ GRv @ v))
    HV = {}  # name -> effective applied head vector (alpha = 1), for cosines and gains
    RV = {}
    # ---- A: IDEEA clustered (nc=2 min_perp) on train rows, + block metric
    XH = HB  # IDEEA's own activations
    cl = {}
    for j, (l, h) in enumerate(HEADS):
        P_, N_ = XH[pos][:, j * HD:(j + 1) * HD], XH[neg][:, j * HD:(j + 1) * HD]
        kp = KMeans(n_clusters=2, random_state=0, n_init=10).fit(P_); kn = KMeans(n_clusters=2, random_state=0, n_init=10).fit(N_)
        pc, nc_ = kp.cluster_centers_, kn.cluster_centers_
        dm = np.zeros((2, 2, HD))
        for i in range(2):
            for k in range(2):
                d = (pc[k] - nc_[i]).astype(np.float32); dm[i][k] = d / np.linalg.norm(d)
        perm, _ = solve_qap_bruteforce(dm, 2)
        cl[(l, h)] = (pc, nc_, perm, dm)
    for rel in LG + [1e6, 1e12]:
        dirs = {}
        for j, (l, h) in enumerate(HEADS):
            pc, nc_, perm, dm = cl[(l, h)]
            if rel == INF:
                dirs[(l, h)] = np.array([dm[i][perm[i]] for i in range(2)])
            else:
                dirs[(l, h)] = np.array([unit(MB[j]((pc[perm[i]] - nc_[i]).astype(np.float32).astype(np.float64), rel)) for i in range(2)])
        save_ideea_format(f"A_{lamname(rel)}", 2, dirs, {k: v[0] for k, v in cl.items()}, {k: v[1] for k, v in cl.items()}, {k: v[2] for k, v in cl.items()})
        if rel in (1e6, 1e12):
            ref = json.load(open(f"{DIRS}/A_inf.json"))["cluster_dirs"]
            res["sanity"][f"A_{lamname(rel)}_min_cos"] = float(min(cos(np.array(dirs[(l, h)][i]), np.array(ref[l][h][i])) for l, h in HEADS for i in range(2)))
        else:
            HV[f"A_{lamname(rel)}_c0"] = np.concatenate([dirs[k][0] * s for k, s in zip(HEADS, STD)])
    # ---- B (per-head) / G (global, full metric): pooled class means
    dmuH = XH[pos].astype(np.float64).mean(0) - XH[neg].astype(np.float64).mean(0)
    for rel in LG + [1e6, 1e12]:
        vB = blockmet(dmuH, rel); save_iti_format(f"B_{lamname(rel)}", [unit(b) for b in blocks(vB)])
        vG = MH(dmuH, rel); save_iti_format(f"G_{lamname(rel)}", gl_encode(vG))
        if rel in (1e6, 1e12):
            res["sanity"][f"B_{lamname(rel)}_cos"] = cos(ph_effective(vB), ph_effective(dmuH)); res["sanity"][f"G_{lamname(rel)}_cos"] = cos(vG, dmuH)
        else:
            HV[f"B_{lamname(rel)}"] = ph_effective(vB); HV[f"G_{lamname(rel)}"] = vG / np.linalg.norm(vG) * S_TOT
    # ITI refit check: B_inf == ITI mass-mean on train rows (pooled) — compare with the dev-fitted ITI for information
    iti = np.array(json.load(open(ITI_PATH))["normalized_dir"])
    res["cos"]["B_inf_vs_ITI_devfit_perhead_mean"] = float(np.mean([cos(b, iti[l, h]) for b, (l, h) in zip(blocks(dmuH), HEADS)]))
    # ---- E: eval-context pooled mean difference (head), J (residual)
    selr = R["is_train"] & ~R["noninfo"]
    cE = R["cH"][selr & R["correct"]].astype(np.float64).mean(0) - R["cH"][selr & ~R["correct"]].astype(np.float64).mean(0)
    save_iti_format("E_inf", [unit(b) for b in blocks(cE)]); HV["E_inf"] = ph_effective(cE)
    cJ = R["cR"][selr & R["correct"]].astype(np.float64).mean(0) - R["cR"][selr & ~R["correct"]].astype(np.float64).mean(0)
    # ---- C (per-head, block metric) / H (global, full metric): gradient of r_fix at the head site
    for rel in LG + [1e6, 1e12]:
        vC = blockmet(gH, rel); save_iti_format(f"C_{lamname(rel)}", [unit(b) for b in blocks(vC)])
        vH = MH(gH, rel); save_iti_format(f"H_{lamname(rel)}", gl_encode(vH))
        if rel in (1e6, 1e12):
            res["sanity"][f"C_{lamname(rel)}_cos"] = cos(ph_effective(vC), ph_effective(gH)); res["sanity"][f"H_{lamname(rel)}_cos"] = cos(vH, gH)
        else:
            HV[f"C_{lamname(rel)}"] = ph_effective(vC); HV[f"H_{lamname(rel)}"] = vH / np.linalg.norm(vH) * S_TOT
    # ---- D: per-question gradients clustered by eval-prompt-end context (NON-IDEEA variant), nearest-centroid selection
    PA = dict(np.load(f"{OUT}/prompt_acts.npz"))
    pidx = {int(q): k for k, q in enumerate(PA["qidx"])}
    Xq = PA["hH"][[pidx[int(q)] for q in gqs]]
    dcl = {}
    for j, (l, h) in enumerate(HEADS):
        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(Xq[:, j * HD:(j + 1) * HD])
        dcl[(l, h)] = (km.cluster_centers_, [gq[km.labels_ == c][:, j * HD:(j + 1) * HD].mean(0) for c in range(2)], np.bincount(km.labels_, minlength=2).tolist())
    res["D_cluster_sizes"] = {f"{l}.{h}": v[2] for (l, h), v in dcl.items()}
    for rel in LG:
        dirs = {k: np.array([unit(MB[j](v[1][c], rel)) for c in range(2)]) for j, (k, v) in enumerate(dcl.items())}
        far = {k: np.full((2, HD), 1e4) for k in dcl}
        save_ideea_format(f"D_{lamname(rel)}", 2, dirs, far, {k: v[0] for k, v in dcl.items()}, {k: [0, 1] for k in dcl})
    # ---- F: kernel bank (unpaired pos / neg rows), lam_rel in {1, inf}
    rs = np.random.RandomState(0)
    bank = {}
    for j, (l, h) in enumerate(HEADS):
        P_, N_ = XH[pos][:, j * HD:(j + 1) * HD].astype(np.float64), XH[neg][:, j * HD:(j + 1) * HD].astype(np.float64)
        Gb = GH[j * HD:(j + 1) * HD, j * HD:(j + 1) * HD]; M = Gb + 0.1 * np.trace(Gb) / HD * np.eye(HD)
        allr = np.concatenate([P_, N_]); ia, ib = rs.randint(0, len(allr), 3000), rs.randint(0, len(allr), 3000)
        df_ = allr[ia] - allr[ib]
        bank[f"P{j}"], bank[f"N{j}"] = P_.astype(np.float32), N_.astype(np.float32)
        bank[f"P{j}_l1"], bank[f"N{j}_l1"] = MB[j](P_.T, 1.0).T.astype(np.float32), MB[j](N_.T, 1.0).T.astype(np.float32)
        bank[f"M{j}"] = M.astype(np.float32)
        bank[f"t{j}_euc"] = float(np.median((df_ ** 2).sum(1))); bank[f"t{j}_mah"] = float(np.median(np.einsum("nd,de,ne->n", df_, M, df_)))
        # limit check (CPU): tau = inf kernel mean diff equals pooled mass-mean (B_inf) and lam=1 version equals B_1
        res["sanity"].setdefault("F_tauinf_vs_B_cos_min", 1.0)
        res["sanity"]["F_tauinf_vs_B_cos_min"] = min(res["sanity"]["F_tauinf_vs_B_cos_min"], cos(P_.mean(0) - N_.mean(0), dmuH[j * HD:(j + 1) * HD]),
                                                    cos(bank[f"P{j}_l1"].mean(0) - bank[f"N{j}_l1"].mean(0), MB[j](dmuH[j * HD:(j + 1) * HD], 1.0)))
    np.savez(f"{DIRS}/F_bank.npz", **bank)
    # n_eff of every finite-tau F config at the prompt-end activations of the TEST questions (what the hook sees at step 1)
    te = PA["split"] == "test"; XT = PA["hH"][te].astype(np.float64)
    neff = {}
    for dist in ("euc", "mah"):
        for trel in (0.05, 0.2):
            per = []
            for j in range(len(HEADS)):
                x = XT[:, j * HD:(j + 1) * HD]; tau = trel * bank[f"t{j}_{dist}"]
                Mj = bank[f"M{j}"].astype(np.float64) if dist == "mah" else np.eye(HD)
                for Bk in (bank[f"P{j}"], bank[f"N{j}"]):
                    Bk = Bk.astype(np.float64); xm, bm = x @ Mj, Bk @ Mj
                    d2 = (xm * x).sum(1)[:, None] + (bm * Bk).sum(1)[None] - 2 * xm @ Bk.T
                    lw = -d2 / tau; lw -= lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
                    per.append(1.0 / (w ** 2).sum(1))
            per = np.array(per)  # (48, n_test)
            m = per.mean(0)
            neff[f"{dist}_{trel:g}"] = {"median_over_test_q": float(np.median(m)), "p10": float(np.percentile(m, 10)), "p90": float(np.percentile(m, 90)),
                                        "bank_sizes": [int(pos.sum()), int(neg.sum())]}
    res["F_neff"] = neff
    # ---- I (residual bare-context pooled md, + metric), J
    dmuR = B_["resid"][pos].astype(np.float64).mean(0) - B_["resid"][neg].astype(np.float64).mean(0)
    resid = {"J_inf": cJ}
    for rel in LG + [1e6, 1e12]:
        resid[f"I_{lamname(rel)}"] = MR(dmuR, rel)
    res["sanity"]["I_1e6_cos"] = cos(resid["I_1e+06"], dmuR); res["sanity"]["I_1e12_cos"] = cos(resid["I_1e+12"], dmuR)
    np.savez(f"{DIRS}/resid.npz", **resid)
    for k, v in resid.items():
        if k not in ("I_1e+06", "I_1e+12"):
            RV[k] = v
    for k in ("efsum", "nfsum0.1", "nfsum1", "nfsum10"):
        RV["K_" + {"efsum": "inf", "nfsum0.1": "0.1", "nfsum1": "1", "nfsum10": "10"}[k]] = RZ[k]
    RV["CAA_AB_L13"] = np.array(json.load(open(os.path.join(ROOT, "results/truthfulqa/gemma2_2b/directions/caa/directions.json")))["direction"][13])
    # original IDEEA (dev-fitted, all refs) cluster dir 0 and ITI-dev at the head site, for cosines
    o2 = json.load(open(IDEEA_NC2))
    HV["L0_IDEEA_dev_c0"] = np.concatenate([np.array(o2["cluster_dirs"][l][h][0]) * s for (l, h), s in zip(HEADS, STD)])
    HV["ITI_dev"] = np.concatenate([iti[l, h] * s for (l, h), s in zip(HEADS, STD)])
    HV["gradH_raw"], HV["dmuH_raw"], HV["ctxH_raw"] = gH, dmuH, cE
    # ---- cosines and first-order val gains of r_fix per unit G-norm (label-free; val refs + val samples)
    hk, rk = sorted(HV), sorted(RV)
    res["cos"]["head"] = {a_: {b_: round(cos(HV[a_], HV[b_]), 4) for b_ in hk} for a_ in hk}
    res["cos"]["head_perhead_mean"] = {a_: {b_: round(float(np.mean([cos(x, y) for x, y in zip(blocks(HV[a_]), blocks(HV[b_]))])), 4) for b_ in hk} for a_ in hk}
    res["cos"]["resid"] = {a_: {b_: round(cos(RV[a_], RV[b_]), 4) for b_ in rk} for a_ in rk}
    res["gain"]["head"] = {k: gainH(v) for k, v in HV.items()}
    res["gain"]["resid"] = {k: gainR(v) for k, v in RV.items()}
    res["hnorm13"] = hn13
    json.dump(res, open(f"{OUT}/fit_bridge.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cos",)}, indent=1)[:6000])
    print("cos head (global):", json.dumps({k: res["cos"]["head"][k] for k in ("B_inf", "C_inf", "E_inf", "H_inf", "H_1", "G_1")}, indent=0)[:3000])
    print("cos resid:", json.dumps(res["cos"]["resid"], indent=0)[:3000])


# ----------------------------------------------------------------------------------------------- eval-side methods
class _Rm:
    def __init__(self, hs):
        self.hs = hs

    def remove(self):
        for h in self.hs:
            h.remove()


class FileITI:
    """IDEEA's own ITI hook (k=24) with a substituted directions file."""

    def __init__(self, path, alpha):
        self.path, self.alpha = path, alpha

    def install(self, model, task, model_name):
        import methods.iti as MI
        from methods.iti import ITI
        old = MI.directions_path
        MI.directions_path = lambda *a, **k: Path(self.path)
        try:
            return ITI(K_HEADS, self.alpha).install(model, task, model_name)
        finally:
            MI.directions_path = old


class FileIDEEA:
    def __init__(self, path, variant, nc, alpha):
        self.path, self.variant, self.nc, self.alpha = path, variant, nc, alpha

    def install(self, model, task, model_name):
        import methods.ideea as MI
        old = MI.directions_path
        MI.directions_path = lambda *a, **k: Path(self.path)
        try:
            return MI.IDEEA(self.variant, MI.FixedClusters(self.nc), K_HEADS, self.alpha).install(model, task, model_name)
        finally:
            MI.directions_path = old


class KernelHeads:
    """Arm F. Per head h and per forward call, x = current last-position head input; mu+(x), mu-(x) = kernel means of the
    (lam-projected) positive / negative train rows with k = exp(-dist^2/tau); v = unit(mu+ - mu-); x += alpha*std_h*v.
    mode 'soft' (tau, dist) or 'hard': one-hot nearest IDEEA centroid (real nc=4 dev-fitted centroids) = IDEEA nearest_pos_neg."""

    def __init__(self, alpha, mode="soft", dist="euc", trel=0.2, lam="inf"):
        self.alpha, self.mode, self.dist, self.trel, self.lam = alpha, mode, dist, trel, lam

    def install(self, model, task, model_name):
        import torch
        from methods.ideea import _select_nearest_pos_neg
        hs = []
        if self.mode == "hard":
            d = json.load(open(IDEEA_NC4)); ns = np.array(d["naive_std"])
            by = {}
            for l, h in HEADS:
                by.setdefault(l, []).append(h)
            for l, hl in by.items():
                pc = {h: torch.tensor(np.array(d["pos_centroids"][l][h]), dtype=torch.float32) for h in hl}
                nc_ = {h: torch.tensor(np.array(d["neg_centroids"][l][h]), dtype=torch.float32) for h in hl}

                def hook(module, inputs, l=l, hl=hl, pc=pc, nc_=nc_):
                    xf = inputs[0]
                    for h in hl:
                        x = xf[:, -1, h * HD:(h + 1) * HD]
                        v = _select_nearest_pos_neg(x, None, pc[h].to(x.device, x.dtype), nc_[h].to(x.device, x.dtype), None)
                        x += self.alpha * v * ns[l][h]
                    return inputs
                hs.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
            return _Rm(hs)
        Z = np.load(f"{DIRS}/F_bank.npz")
        sfx = "" if self.lam == "inf" else "_l1"
        T = {}
        for j, (l, h) in enumerate(HEADS):
            T[j] = {k: torch.tensor(Z[f"{k}{j}"], device="cuda") for k in ("P", "N")}
            T[j].update({k + "v": torch.tensor(Z[f"{k}{j}{sfx}"], device="cuda") for k in ("P", "N")})
            T[j]["M"] = torch.tensor(Z[f"M{j}"], device="cuda")
            T[j]["tau"] = self.trel * float(Z[f"t{j}_{self.dist}"]) if self.trel != "inf" else None
        by = {}
        for j, (l, h) in enumerate(HEADS):
            by.setdefault(l, []).append((h, j))
        for l, hl in by.items():
            def hook(module, inputs, hl=hl):
                xf = inputs[0]
                for h, j in hl:
                    t = T[j]
                    x = xf[:, -1, h * HD:(h + 1) * HD]
                    xs = x.float()
                    mus = []
                    for k in ("P", "N"):
                        Bk, Bv = t[k], t[k + "v"]
                        if t["tau"] is None:
                            mus.append(Bv.mean(0)[None].expand(xs.shape[0], -1))
                            continue
                        if self.dist == "euc":
                            d2 = (xs ** 2).sum(-1, keepdim=True) + (Bk ** 2).sum(-1)[None] - 2 * xs @ Bk.T
                        else:
                            xm = xs @ t["M"]; bm = Bk @ t["M"]
                            d2 = (xm * xs).sum(-1, keepdim=True) + (bm * Bk).sum(-1)[None] - 2 * xm @ Bk.T
                        w = torch.softmax(-d2 / t["tau"], -1)
                        mus.append(w @ Bv)
                    v = mus[0] - mus[1]; v = v / v.norm(dim=-1, keepdim=True)
                    x += (self.alpha * STD[j] * v).to(x.dtype)
                return inputs
            hs.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
        return _Rm(hs)


class GateSteer:
    """Arm O: residual L13 output += mult * hnorm * delta(x); delta(x) = Z[0] (K=1) or softmax(W x/||x||) @ Z.
    mode 'from' (all positions of the call; teacher forcing) or 'last' (generation protocol)."""

    def __init__(self, model, Z, W=None, hn=1.0, layer=RL):
        self.Z, self.W, self.hn, self.mult, self.mode, self.on = Z, W, hn, 1.0, "last", True
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def delta(self, x):
        import torch
        if self.W is None:
            return self.hn * self.Z[0].to(x.dtype).expand_as(x)
        g = torch.softmax((x.float() / x.float().norm(dim=-1, keepdim=True)) @ self.W.float().T, -1)
        return (self.hn * g @ self.Z.float()).to(x.dtype)

    def _hook(self, m, i, o):
        if not self.on:
            return o
        h = o[0] if isinstance(o, tuple) else o
        if self.mode == "last":
            add = self.delta(h[:, -1:]) * self.mult
            h2 = torch_cat_last(h, add)
        else:
            h2 = h + self.mult * self.delta(h)
        return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2

    def remove(self):
        self.handle.remove()


def torch_cat_last(h, add):
    import torch
    return torch.cat([h[:, :-1], h[:, -1:] + add], 1)


class GateMethod:
    def __init__(self, name, c):
        self.name, self.c = name, c

    def install(self, model, task, model_name):
        import torch
        z = np.load(f"{DIRS}/{self.name}.npz")
        Z = torch.tensor(z["Z"], device="cuda"); W = torch.tensor(z["W"], device="cuda") if "W" in z.files else None
        gs = GateSteer(model, Z, W, float(z["hn"]))
        gs.mult = self.c / float(np.mean(np.linalg.norm(z["Z"], axis=1)))  # strength c = ||delta_k|| / hnorm (mean over k)
        return _Rm([gs.handle])


def build_bridge(ctx, spec):
    p = spec.split(":"); name, s = p[1], float(p[2])
    fam = name.split("_")[0]
    info = {"method": "br", "name": name, "strength": s}
    if fam in ("B", "C", "E", "G", "H"):
        return ("ideea_hooks", FileITI(f"{DIRS}/{name}.json", s)), info
    if fam == "A":
        return ("ideea_hooks", FileIDEEA(f"{DIRS}/{name}.json", "min_perp", 2, s)), info
    if fam == "D":
        return ("ideea_hooks", FileIDEEA(f"{DIRS}/{name}.json", "nearest_cluster", 2, s)), info
    if fam == "Aorig":  # original dev-fitted IDEEA nc2 file through the file mechanism (wiring check)
        return ("ideea_hooks", FileIDEEA(IDEEA_NC2, "min_perp", 2, s)), info
    if fam == "F":  # F_hard | F_inf_inf | F_{euc|mah}{0.05|0.2}_{inf|1}
        if name == "F_hard":
            return ("ideea_hooks", KernelHeads(s, "hard")), info
        a_, lam = name.split("_")[1:3]
        if a_ == "inf":
            return ("ideea_hooks", KernelHeads(s, "soft", "euc", "inf", lam)), info
        return ("ideea_hooks", KernelHeads(s, "soft", a_[:3], float(a_[3:]), lam)), info
    if fam in ("I", "J"):
        v = np.load(f"{DIRS}/resid.npz")[name]
        th = v / np.linalg.norm(v) * s * ctx.hnorm(RL)
        return ("fixed", th, RL), info
    if fam.startswith("O"):
        return ("ideea_hooks", GateMethod(name, s)), info
    raise ValueError(spec)


def patch_eval():
    import gm_eval as E
    orig = E.build

    def build(ctx, spec):
        return build_bridge(ctx, spec) if spec.startswith("br:") else orig(ctx, spec)
    E.build = build
    return E


HEAD_GRID = [0.5, 1, 2, 4, 8, 16]
RES_GRID = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
grid_for = lambda name: RES_GRID if name.split("_")[0] in ("I", "J") or name.startswith("O") else HEAD_GRID


def stage_eval(a):
    import torch
    import ng_lib as L
    E = patch_eval()
    ctx = E.Ctx(a.split)
    if a.max_q:
        ctx.requests = ctx.requests[:a.max_q]
    tok, model = L.load_model(torch.bfloat16)
    od = os.path.join(ROOT, "results_gm", a.split + "_" + a.tag); os.makedirs(od, exist_ok=True)
    steers = {}
    specs = list(a.specs)
    for nm in a.calib:
        specs += [f"br:{nm}:{c:g}" for c in grid_for(nm)]
    for s in specs:
        E.run_spec(ctx, model, tok, steers, s, 8, od)


def c_for(name, targets, tag="bridge"):
    od = os.path.join(ROOT, "results_gm", "development_calib" + tag)
    pts = [(c, json.load(open(f"{od}/br_{name}_{c:g}.json"))["summary"]["kl_tok_mean"]) for c in grid_for(name)]
    out = []
    for T in targets:
        cc = None
        for (c0, k0), (c1, k1) in zip(pts[:-1], pts[1:]):
            if k0 <= T <= k1 and k1 > k0:
                cc = float(np.exp(np.log(c0) + (np.log(T) - np.log(max(k0, 1e-5))) / (np.log(k1) - np.log(max(k0, 1e-5))) * (np.log(c1) - np.log(c0)))); break
        if cc is None:
            (c0, k0), (c1, k1) = (pts[0], pts[1]) if T < pts[0][1] else (pts[-2], pts[-1])
            sl = (np.log(max(k1, 1e-5)) - np.log(max(k0, 1e-5))) / (np.log(c1) - np.log(c0))
            cc = float(np.clip(c1 * np.exp((np.log(T) - np.log(max(k1, 1e-5))) / max(sl, 0.5)), 0.01, 64.0))
        out.append(round(cc, 4))
    return out, pts


# ----------------------------------------------------------------------------------------------- arm O (optimised)
def stage_opt(a):
    import torch
    import ng_lib as L
    from gm_fit import ref_conts, chat_for
    from gm_fix import NONINFO
    torch.manual_seed(0)
    df = pd.read_csv(CSV)
    tok, model = L.load_model(torch.float32)
    d = model.config.hidden_size
    hn = json.load(open(f"{OUT}/fit_bridge.json"))["hnorm13"]
    K = a.K
    Z = torch.nn.Parameter(torch.zeros(K, d, device="cuda") if K == 1 else 1e-3 * torch.randn(K, d, device="cuda"))
    W = torch.nn.Parameter(1e-3 * torch.randn(K, d, device="cuda")) if K > 1 else None
    gs = GateSteer(model, Z, W, hn); gs.mode = "from"
    opt = torch.optim.Adam([Z] + ([W] if W is not None else []), lr=3e-3)
    S = {sp: dict(np.load(f"{OUT}/samples_{sp}.npz")) for sp in ("train", "val")}
    data = {}
    for sp in ("train", "val"):
        for i in split_of(sp):
            q = df.loc[i, "Question"]; pid = L.prompt_ids(tok, chat_for(q))
            good = [x for x in split_refs(df.loc[i, "Correct Answers"]) if not NONINFO.match(x)]
            bad = [x for x in split_refs(df.loc[i, "Incorrect Answers"]) if not NONINFO.match(x)]
            if not good or (not bad and a.obj == "contrast"):
                continue
            conts, _ = ref_conts(tok, pid, q, good + bad)
            m = np.where(S[sp]["qidx"] == i)[0]
            samp = [torch.tensor(S[sp]["toks"][S[sp]["offs"][k]:S[sp]["offs"][k + 1]]) for k in m]
            data[(sp, i)] = (pid, conts, len(good), samp)

    def qloss(key, grad):
        pid, conts, ng, samp = data[key]
        pid = pid.cuda()
        gs.on = False
        kv = L._prefix_cache(model, pid)  # prompt positions 0..P-2 are never steered
        cs = [c.cuda() for c in (conts if a.obj == "contrast" else conts[:ng])]
        nb = len(cs) - ng
        wts = [1.0 / ng] * ng + ([-1.0 / nb] * nb if a.obj == "contrast" else [])
        Jv = 0.0
        for b in range(0, len(cs), 6):  # J is linear in the per-ref log-probs -> chunked backward
            gs.on = True
            lg, tgt, tm = L.cont_logits(model, gs_proxy, pid, cs[b:b + 6], None, kv)
            lp = (torch.log_softmax(lg, -1).gather(-1, tgt[..., None])[..., 0] * tm).sum(1)
            Jb = (lp * torch.tensor(wts[b:b + 6], device=lp.device)).sum()
            if grad:
                (-Jb).backward()
            Jv += float(Jb.detach()); del lg, lp, Jb
        ntot = float(sum(c.numel() for c in samp)); KLv = 0.0
        for b in range(0, len(samp), 4):
            ss = [c.cuda() for c in samp[b:b + 4]]
            gs.on = False
            with torch.no_grad():
                l0, _, _ = L.cont_logits(model, gs_proxy, pid, ss, None, kv)
                lp0 = torch.log_softmax(l0, -1); del l0
            gs.on = True
            ls, _, tms = L.cont_logits(model, gs_proxy, pid, ss, None, kv)
            lps = torch.log_softmax(ls, -1); del ls
            klt = (lps.exp() * (lps - lp0)).sum(-1)
            KLb = (klt * tms).sum() / ntot
            if grad:
                (a.beta * KLb).backward()
            KLv += float(KLb); del lps, lp0, klt, KLb
        return Jv, KLv

    class _P:  # cont_logits sets .mode/.start/.theta on its steer arg; GateSteer ignores theta
        mode = start = theta = None
    gs_proxy = _P()
    trk = [k for k in data if k[0] == "train"]; vak = [k for k in data if k[0] == "val"]
    rs = np.random.RandomState(0)
    hist, best = [], None
    nstep = int(np.ceil(len(trk) / 8)); half = max(1, nstep // 2)
    tag = a.name
    t0 = time.time()
    for ep in range(3):
        order = rs.permutation(len(trk))
        for st in range(nstep):
            opt.zero_grad()
            for k in order[st * 8:(st + 1) * 8]:
                qloss(trk[k], True)
            for p_ in [Z] + ([W] if W is not None else []):
                p_.grad /= 8
            opt.step()
            if (st + 1) % half == 0 or st == nstep - 1:
                with torch.no_grad():
                    vals = [qloss(k, False) for k in vak]
                vJ, vK = float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))
                obj = vJ - a.beta * vK
                hist.append({"epoch": ep, "step": st, "val_J": vJ, "val_KL": vK, "val_obj": obj, "norm_over_hn": float(Z.norm(dim=1).mean()), "t": time.time() - t0})
                print(json.dumps(hist[-1]), flush=True)
                if best is None or obj > best[0]:
                    best = (obj, Z.detach().cpu().numpy().copy(), None if W is None else W.detach().cpu().numpy().copy(), len(hist) - 1)
    out = {"Z": best[1], "hn": hn}
    if best[2] is not None:
        out["W"] = best[2]
    np.savez(f"{DIRS}/{tag}.npz", **out)
    json.dump({"name": tag, "K": K, "obj": a.obj, "beta": a.beta, "best_checkpoint": best[3], "hist": hist,
               "cos_with_efsum": cos(best[1].mean(0), np.load(f"{ROOT}/runs_ng/directions_fix_L13.npz")["efsum"]),
               "cos_with_nfsum10": cos(best[1].mean(0), np.load(f"{ROOT}/runs_ng/directions_fix_L13.npz")["nfsum10"])},
              open(f"{OUT}/opt_{tag}.json", "w"), indent=1)
    gs.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "fit", "eval", "cfor", "opt"])
    ap.add_argument("--specs", nargs="*", default=[]); ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--split", default="development"); ap.add_argument("--tag", default="bridge"); ap.add_argument("--max_q", type=int, default=0)
    ap.add_argument("--names", nargs="*", default=[]); ap.add_argument("--targets", nargs="*", type=float, default=[0.1, 0.3, 0.6, 1.0])
    ap.add_argument("--K", type=int, default=1); ap.add_argument("--obj", default="contrast"); ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--name", default="O1")
    a = ap.parse_args()
    if a.stage == "cfor":
        specs = []
        for nm in a.names:
            cs_, pts = c_for(nm, a.targets)
            print(f"# {nm} scan {[(c, round(x, 4)) for c, x in pts]} -> {cs_}", file=sys.stderr)
            specs += [f"br:{nm}:{c:g}" for c in cs_]
        print(" ".join(specs))
    else:
        {"feats": stage_feats, "fit": stage_fit, "eval": stage_eval, "opt": stage_opt}[a.stage](a)

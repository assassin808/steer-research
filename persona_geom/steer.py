"""Stage S: steering (prereg section 6). Layer index l = hidden_states index (l = output of block l-1); steering at l hooks block l-1.

usage: RUN=main python steer.py sweep     layer choice per trait: persona vector at LCAND, 10 TRAIN own-source questions x 2 samples
       RUN=main python steer.py calib     at l*_t: fit arms (incl. per-token IDEEA states), per arm the smallest strength reaching
                                          the prompt-strong target score on TRAIN questions with coherence >= neutral - 1
       RUN=main python steer.py eval      TEST questions x arms x 8 strengths x S_STEER samples; 6 judges + coherence + per-token KL
Inputs: <OUT>/judged_gen_main.json, <OUT>/acts_neu.npy.   Outputs: <OUT>/steer_{sweep,calib,eval}.json, <OUT>/arms_<trait>.npz
Arms: pv, chord, geo, geo_null (control), ideea1, ideea; prompt arm comes from gen_main (analyze_steer.py).
"""
import os, sys, json, numpy as np, torch
from common import OUT, SMOKE, TRAITS, MAXNEW, jload, jdump, log, seed_of, questions
from lm import LM, kl_tf, token_states
from judge import Judge, score_items
from paths import fit_arms, delta, labels

STAGE = sys.argv[1]
LCAND = [int(x) for x in os.environ.get("LCAND", "12" if SMOKE else "6,9,12,15,18").split(",")]
MULT = [1, 2, 4, 8]                                  # sweep multipliers of the raw persona vector
CAL = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12]               # calibration strengths (multipliers of the arm's unit displacement)
NSTR = 8                                              # eval strengths: fractions 0, 1/7, ..., 1 of the calibrated max
S_STEER = int(os.environ.get("S_STEER", "4"))         # TBD from power curve
Q_STEER_OWN, Q_STEER_OTHER = 20, 10
COH_DROP = 1.0
ADD_ARMS = ["pv", "chord", "geo", "geo_null"]
IDEEA_ARMS = ["ideea1", "ideea"]


def load():
    rows = jload(f"{OUT}/judged_gen_main.json")
    meta = {k: np.array([r[k] for r in rows]) for k in ["trait", "level", "sample", "split", "src", "qid"]}
    qidx = {v: i for i, v in enumerate(sorted(set(meta["qid"])))}; meta["q"] = np.array([qidx[v] for v in meta["qid"]])
    for t in TRAITS: meta[f"P_{t}"] = np.array([r[f"P_{t}"] for r in rows])
    meta["coh"] = np.array([r["coh"] for r in rows])
    A = np.load(f"{OUT}/acts_neu.npy", mmap_mode="r")
    return rows, meta, A


def spec_for(arm, arms, f, cal):
    if arm == "pv": return ("add", delta("pv", arms, alpha=f * cal))
    if arm in ("chord", "geo", "geo_null"): return ("add", delta(arm, arms, s=f, lam=cal))
    pre = "ideea1" if arm == "ideea1" else "ideea"
    return ("ideea", (arms[pre + "_pos"], arms[pre + "_neg"], f * cal))


def steer_set(lm, L, specs, from_pos=None):
    """specs: per row ('add', delta) | ('ideea', (pos, neg, alpha)) | None. Rows of one call share mode and centroids."""
    def f(ch):
        sp = [specs[i] for i in ch]
        if all(s is None for s in sp): lm.clear(); return
        kinds = {s[0] for s in sp if s is not None}; assert len(kinds) == 1
        if "add" in kinds:
            lm.set_add(L - 1, [None if s is None else s[1] for s in sp])
        else:
            s0 = next(s for s in sp if s is not None)
            lm.set_ideea(L - 1, s0[1][0], s0[1][1], [0.0 if s is None else s[1][2] for s in sp],
                         from_pos=None if from_pos is None else from_pos[ch])
    return f


def run(lm, J, items, L, tag, want_kl=True):
    """items: dicts with q, spec, group (rows of one group share hook mode + centroids). Neutral context."""
    out = []
    groups = sorted({it["group"] for it in items})
    for g in groups:
        sub = [it for it in items if it["group"] == g]
        specs = [it["spec"] for it in sub]
        ctx = [lm.chat_ids(None, it["q"]) for it in sub]
        resps = lm.generate(ctx, seed_of(tag, g), MAXNEW, steer=steer_set(lm, L, specs))
        if want_kl and any(s is not None for s in specs):
            fr = torch.tensor([len(c) - 1 for c in ctx], device=lm.model.device)
            kl = kl_tf(lm, ctx, resps, steer_set(lm, L, specs, from_pos=fr))
        else:
            kl = [0.0] * len(sub)
        for it, r, k in zip(sub, resps, kl):
            o = {kk: v for kk, v in it.items() if kk not in ("spec", "group")}
            o.update(text=lm.decode(r), ntok=len([t for t in r if t not in lm.stop]), natural_end=bool(r and r[-1] in lm.stop), kl_tok=k)
            out.append(o)
    sc = score_items(J, out)
    for k, v in sc.items():
        for o, x in zip(out, v): o[k] = x
    return out


def own_questions(split, t, n):
    return [q for q in questions() if q["split"] == split and q["src"] == t][:n]


def steer_questions(t, rng):
    own = own_questions("test", t, 2 if SMOKE else Q_STEER_OWN)
    oth = [q for q in questions() if q["split"] == "test" and q["src"] != t]
    pick = rng.choice(len(oth), 1 if SMOKE else Q_STEER_OTHER, replace=False)
    return own + [oth[i] for i in sorted(pick)]


def prompt_target(meta, t, qids):
    m = (meta["trait"] == t) & (meta["level"] == 3) & np.isin(meta["qid"], qids)
    return float(meta[f"P_{t}"][m].mean())


def ideea_states(lm, rows, meta, t, L, train):
    """raw per-token residuals at l of the D+_t / D-_t TRAIN responses, re-read in the neutral context (as acts_neu)"""
    Dp, Dn = labels(meta, t); out = []
    for D in (Dp & train, Dn & train):
        idx = np.where(D)[0]
        if len(idx) > 400: idx = np.random.default_rng(0).choice(idx, 400, replace=False)
        out.append(token_states(lm, [lm.chat_ids(None, rows[i]["q"]) for i in idx], [rows[i]["resp"] for i in idx], L))
    return out


def save_arms(t, arms):
    np.savez(f"{OUT}/arms_{t}.npz", **{k: v for k, v in arms.items() if isinstance(v, np.ndarray)},
             meta=json.dumps({k: v for k, v in arms.items() if not isinstance(v, np.ndarray)}))


def load_arms(t):
    z = np.load(f"{OUT}/arms_{t}.npz"); arms = {k: z[k] for k in z.files if k != "meta"}; arms.update(json.loads(str(z["meta"])))
    return arms


if __name__ == "__main__":
    rows, meta, A = load()
    train = meta["split"] == "train"
    lm, J = LM(), Judge()
    ns = 1 if SMOKE else 2
    if STAGE == "sweep":
        res = []
        for t in TRAITS:
            qs = own_questions("train", t, 2 if SMOKE else 10)
            for L in LCAND:
                arms = fit_arms(np.asarray(A[:, L], np.float64), meta, t, train)
                items = [dict(trait=t, L=L, mult=mu, qid=q["qid"], q=q["q"], sample=s, group="add",
                              spec=("add", delta("pv", arms, alpha=mu)) if mu else None)
                         for mu in [0] + MULT for q in qs for s in range(ns)]
                res += run(lm, J, items, L, f"sweep{t}{L}", want_kl=False)
                log("sweep", t, L, "|pv|", float(np.linalg.norm(arms["pv"])))
            jdump(res, f"{OUT}/steer_sweep.json")
        best = {}
        for t in TRAITS:
            R = [r for r in res if r["trait"] == t]; coh0 = np.mean([r["coh"] for r in R if r["mult"] == 0]); sc = {}
            for L in LCAND:
                ok = [np.mean([r[f"P_{t}"] for r in R if r["L"] == L and r["mult"] == mu]) for mu in MULT
                      if np.mean([r["coh"] for r in R if r["L"] == L and r["mult"] == mu]) >= coh0 - COH_DROP]
                sc[L] = max(ok) if ok else -1
            best[t] = int(min(LCAND, key=lambda L: (-sc[L], L))); log("layer", t, best[t], sc)
        jdump(best, f"{OUT}/layers.json")
    elif STAGE == "calib":
        best = jload(f"{OUT}/layers.json"); res, cal = [], {}
        if os.path.exists(f"{OUT}/calib.json"): cal = jload(f"{OUT}/calib.json"); res = jload(f"{OUT}/steer_calib.json")
        resp_rows = jload(f"{OUT}/gen_main.json")    # has token ids
        for t in TRAITS:
            if t in cal: continue
            L = best[t]
            tp, tn = ideea_states(lm, resp_rows, meta, t, L, train)
            arms = fit_arms(np.asarray(A[:, L], np.float64), meta, t, train, tok_pos=tp, tok_neg=tn); save_arms(t, arms)
            qs = own_questions("train", t, 2 if SMOKE else 10)
            tgt = prompt_target(meta, t, [q["qid"] for q in qs])
            items = [dict(trait=t, L=L, strength=0, arm="none", qid=q["qid"], q=q["q"], sample=s, group="add", spec=None)
                     for q in qs for s in range(ns)]
            for c in CAL:
                for q in qs:
                    for s in range(ns):
                        base = dict(trait=t, L=L, strength=c, qid=q["qid"], q=q["q"], sample=s)
                        for arm in ("pv", "chord", "ideea1", "ideea"):        # chord(s=1) == geo(s=1); geo_null uses chord's lam
                            items.append(dict(base, arm=arm, group=arm if arm.startswith("ideea") else "add", spec=spec_for(arm, arms, 1.0, c)))
            R = run(lm, J, items, L, f"calib{t}", want_kl=False); res += R
            coh0 = np.mean([r["coh"] for r in R if r["arm"] == "none"]); cal[t] = dict(L=L, target=tgt, coh0=float(coh0))
            for arm in ("pv", "chord", "ideea1", "ideea"):
                m = {c: (np.mean([r[f"P_{t}"] for r in R if r["arm"] == arm and r["strength"] == c]),
                         np.mean([r["coh"] for r in R if r["arm"] == arm and r["strength"] == c])) for c in CAL}
                okc = [c for c in CAL if m[c][1] >= coh0 - COH_DROP]; hit = [c for c in okc if m[c][0] >= tgt]
                cal[t][arm] = float(min(hit) if hit else (max(okc) if okc else CAL[0]))
                cal[t][arm + "_reached"] = bool(hit); cal[t][arm + "_curve"] = {str(c): [float(x) for x in m[c]] for c in CAL}
            cal[t]["geo"] = cal[t]["geo_null"] = cal[t]["chord"]
            log("calib", t, {k: v for k, v in cal[t].items() if not k.endswith("_curve")},
                "path straightness data/null", arms["path_straightness"], arms["path_straightness_null"])
            jdump(res, f"{OUT}/steer_calib.json"); jdump(cal, f"{OUT}/calib.json")
    elif STAGE == "eval":
        cal = jload(f"{OUT}/calib.json"); res = []
        if os.path.exists(f"{OUT}/steer_eval.json"): res = jload(f"{OUT}/steer_eval.json")
        done = {r["trait"] for r in res}
        for t in TRAITS:
            if t in done: continue
            L = cal[t]["L"]; arms = load_arms(t)
            qs = steer_questions(t, np.random.default_rng(seed_of("steerq", t)))
            S = 1 if SMOKE else S_STEER
            items = [dict(trait=t, L=L, arm="none", frac=0.0, qid=q["qid"], q=q["q"], sample=s, group="add", spec=None)
                     for q in qs for s in range(S)]
            for k in range(1, NSTR):
                f = k / (NSTR - 1)
                for q in qs:
                    for s in range(S):
                        base = dict(trait=t, L=L, frac=f, qid=q["qid"], q=q["q"], sample=s)
                        for arm in ADD_ARMS + IDEEA_ARMS:
                            items.append(dict(base, arm=arm, group=arm if arm in IDEEA_ARMS else "add", spec=spec_for(arm, arms, f, cal[t][arm])))
            res += run(lm, J, items, L, f"eval{t}", want_kl=True)
            jdump(res, f"{OUT}/steer_eval.json"); log("eval done", t, len(res))

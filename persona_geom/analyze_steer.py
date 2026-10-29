"""H5 steering comparison at MATCHED TARGET-TRAIT GAIN (prereg section 6.4). CPU.
usage: RUN=main python analyze_steer.py  -> <OUT>/steer_summary.json
Arms: prompt (bare neutral + slight/moderate/strong instruction cells from gen_main, same questions), pv, chord, geo,
geo_null (control: same path construction on line-null data), ideea1 (K = 1, last-position rule), ideea (K = 4).
Per arm and trait: curve over strength of target score T, leakage (other traits' score increase), coherence, per-token KL.
Matched gains G_j = j/4 * G_common, j = 1..3, G_common = min(prompt strong gain, every arm's max coherent gain).
Bootstrap CIs are conditional on both arms reaching G_j; the share of replicates dropped is reported (n_boot_used).
Values at G_j by linear interpolation at the first crossing of the gain curve. Bootstrap over questions (paired across arms).
"""
import numpy as np
from common import OUT, TRAITS, NEIGHBOUR_PAIRS, jload, jdump, log

ARMS = ["pv", "chord", "geo", "geo_null", "ideea1", "ideea"]
NB = {t: sorted({b for a, b in NEIGHBOUR_PAIRS if a == t} | {a for a, b in NEIGHBOUR_PAIRS if b == t}) for t in TRAITS}
COH_DROP = 1.0


def curves(R, t, qids):
    """R rows of one trait; returns per arm: x (strength frac), and matrices [n_strength] of means over the given question list"""
    out = {}
    base = [r for r in R if r["arm"] == "none" and r["qid"] in qids]
    for arm in ARMS:
        xs = sorted({r["frac"] for r in R if r["arm"] == arm})
        rows = [base] + [[r for r in R if r["arm"] == arm and r["frac"] == f and r["qid"] in qids] for f in xs]
        out[arm] = dict(x=[0.0] + xs, **_agg(rows, t))
    return out


def _agg(rows_list, t):
    def m(rs, k): return float(np.mean([r[k] for r in rs])) if rs else np.nan
    o = dict(T=[m(rs, f"P_{t}") for rs in rows_list], coh=[m(rs, "coh") for rs in rows_list],
             kl=[m(rs, "kl_tok") if rs and "kl_tok" in rs[0] else 0.0 for rs in rows_list])
    for u in TRAITS:
        if u != t: o[f"P_{u}"] = [m(rs, f"P_{u}") for rs in rows_list]
    return o


def prompt_curve(G, t, qids):
    rows_list = [[r for r in G if r["trait"] == "none" and r["qid"] in qids]] + \
                [[r for r in G if r["trait"] == t and r["level"] == l and r["qid"] in qids] for l in (1, 2, 3, 4)]
    o = dict(x=[0, 1, 2, 3, 4], **_agg(rows_list, t)); o["kl"] = [np.nan] * 5      # prompting has no KL to the unsteered model here
    return o


def leak(c, t, i):
    d = [c[f"P_{u}"][i] - c[f"P_{u}"][0] for u in TRAITS if u != t]
    dn = [c[f"P_{u}"][i] - c[f"P_{u}"][0] for u in NB[t]]
    return dict(mean=float(np.mean(d)), max=float(np.max(d)), nbr=float(np.mean(dn)) if dn else np.nan)


def at_gain(c, t, G):
    """interpolate leakage / coherence / KL at target gain G (first crossing among coherent points); None if not reached"""
    T = np.array(c["T"]); g = T - T[0]; coh0 = c["coh"][0]
    for i in range(1, len(g)):
        if c["coh"][i] < coh0 - COH_DROP: return None
        if g[i] >= G:
            w = (G - g[i - 1]) / (g[i] - g[i - 1]) if g[i] > g[i - 1] else 1.0
            L0, L1 = leak(c, t, i - 1), leak(c, t, i)
            return dict(**{k: (1 - w) * L0[k] + w * L1[k] for k in L0}, coh=(1 - w) * c["coh"][i - 1] + w * c["coh"][i],
                        kl=(1 - w) * c["kl"][i - 1] + w * c["kl"][i], strength=(1 - w) * c["x"][i - 1] + w * c["x"][i])
    return None


def max_coherent_gain(c):
    T = np.array(c["T"]); coh0 = c["coh"][0]; best = 0.0
    for i in range(1, len(T)):
        if c["coh"][i] < coh0 - COH_DROP: break
        best = max(best, T[i] - T[0])
    return best


def max_jump(c):
    T = np.array(c["T"]); coh0 = c["coh"][0]
    k = next((i for i in range(1, len(T)) if c["coh"][i] < coh0 - COH_DROP), len(T))
    T = T[:k]
    tot = abs(T[-1] - T[0]) if len(T) > 1 else 0
    return float(np.max(np.abs(np.diff(T))) / tot) if tot > 0.05 else np.nan


def summarise(R, Gm, t, qids):
    C = curves(R, t, qids); C["prompt"] = prompt_curve(Gm, t, qids)
    Gp = C["prompt"]["T"][3] - C["prompt"]["T"][0]
    Gc = min([Gp] + [max_coherent_gain(C[a]) for a in ARMS])
    out = dict(G_common=float(Gc), G_prompt=float(Gp),
               max_gain={a: float(max_coherent_gain(C[a])) for a in ARMS}, jump={a: max_jump(C[a]) for a in ARMS}, matched={})
    for j in (1, 2, 3):
        G = j / 4 * Gc
        out["matched"][str(j)] = {a: at_gain(C[a], t, G) for a in ARMS + ["prompt"]} if Gc > 0.05 else None
    return out, C


if __name__ == "__main__":
    R_all = jload(f"{OUT}/steer_eval.json"); Gm = jload(f"{OUT}/judged_gen_main.json")
    res = {}
    for t in TRAITS:
        R = [r for r in R_all if r["trait"] == t]
        if not R: continue
        qids = sorted({r["qid"] for r in R})
        point, C = summarise(R, Gm, t, qids)
        rng = np.random.default_rng(0); boots = []
        byq = {}
        for r in R: byq.setdefault(r["qid"], []).append(r)
        gq = {}
        for r in Gm:
            if r["qid"] in byq: gq.setdefault(r["qid"], []).append(r)
        for b in range(500):
            pick = rng.choice(qids, len(qids))
            Rb, Gb, ql = [], [], []
            for i, q in enumerate(pick):  # relabel duplicates so they count separately
                Rb += [dict(r, qid=f"{q}#{i}") for r in byq[q]]; Gb += [dict(r, qid=f"{q}#{i}") for r in gq.get(q, [])]; ql.append(f"{q}#{i}")
            s, _ = summarise(Rb, Gb, t, set(ql))
            boots.append(s)
        diffs = {}
        for j in ("1", "2", "3"):
            for a, b in (("geo", "chord"), ("geo_null", "chord"), ("geo", "geo_null"), ("geo", "pv"), ("geo", "ideea"), ("chord", "pv"),
                         ("ideea", "ideea1"), ("ideea1", "pv"), ("pv", "prompt"), ("geo", "prompt"), ("ideea", "prompt"), ("chord", "prompt")):
                for k in ("mean", "nbr", "max", "coh") + (() if "prompt" in (a, b) else ("kl",)):
                    v = [s["matched"][j][a][k] - s["matched"][j][b][k] for s in boots
                         if s["matched"] and s["matched"][j] and s["matched"][j][a] and s["matched"][j][b]]
                    if len(v) > 50:
                        diffs[f"{a}-{b}|G{j}|{k}"] = dict(mean=float(np.mean(v)), ci=[float(x) for x in np.percentile(v, [2.5, 97.5])], n_boot_used=len(v))
        for a in ("geo", "geo_null"):
            v = [s["jump"][a] - s["jump"]["chord"] for s in boots if np.isfinite(s["jump"][a]) and np.isfinite(s["jump"]["chord"])]
            if v: diffs[f"jump {a}-chord"] = dict(mean=float(np.mean(v)), ci=[float(x) for x in np.percentile(v, [2.5, 97.5])], n_boot_used=len(v))
        jd = [s["jump"]["geo"] - s["jump"]["chord"] for s in boots if np.isfinite(s["jump"]["geo"]) and np.isfinite(s["jump"]["chord"])]
        if jd: diffs["jump geo-chord"] = dict(mean=float(np.mean(jd)), ci=[float(x) for x in np.percentile(jd, [2.5, 97.5])])
        res[t] = dict(point=point, diffs=diffs, curves=C)
        log(t, "G_common", round(point["G_common"], 3), {k: v["ci"] for k, v in diffs.items() if "G2|mean" in k})
    jdump(res, f"{OUT}/steer_summary.json")

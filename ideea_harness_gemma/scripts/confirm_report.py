"""[harness addition] confirm-grad report (prereg results/confirm_grad_prereg.md §4-5).
Inputs: confirm_specs.json, generations + .judged.json, blind_confirm/{a,b,a_rt,b_rt}/labels_*.json + blind_confirm_keys/, results_gm/mc_confirm.
Output: confirm_report.json + printed tables. Runs with whatever labels exist (partial reports)."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_report import degen, kappa  # noqa
from gm_fix_select import is_ref  # noqa

B_ = 10000
S = json.load(open(ROOT + "/confirm_specs.json"))
RUNS = S["runs"]
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith("A:") else t.strip())
NC = re.compile(r"no comment", re.I)


def bci(x, seed=0):
    x = np.asarray(x, float); n = len(x)
    rs = np.random.RandomState(seed); m = np.empty(B_)
    for k in range(0, B_, 1000):
        m[k:k + 1000] = x[rs.randint(0, n, (1000, n))].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "p": float(min(1.0, p))}


gen = {}
for k, v in RUNS.items():
    p = ROOT + "/" + v["path"]
    if not os.path.exists(p):
        continue
    g = json.load(open(p)); rec = {str(r["key"]): r for r in g["records"]}
    gen[k] = rec
QS = sorted(gen["base|mid"], key=int)
FRESH = set(QS) - set(json.load(open(ROOT + "/blind_fix_keys/key_pass1.json"))["questions"])
fresh_mask = np.array([q in FRESH for q in QS])


def per_item(k):
    rec = gen[k]; txt = [full(rec[q]["response"]) for q in QS]
    d = {"dneg_rx": np.array([degen(t)[0] for t in txt], float), "abst_full": np.array([is_ref(t) for t in txt], float),
         "nocomment_any": np.array([bool(NC.search(t)) for t in txt], float), "words": np.array([len(t.split()) for t in txt], float),
         "hitlim": np.array([rec[q]["ntok"] >= 50 for q in QS], float), "kl": np.array([rec[q]["kl_tok"] for q in QS])}
    jp = ROOT + "/" + RUNS[k]["path"].replace(".json", ".judged.json")
    if os.path.exists(jp):
        jd = json.load(open(jp))["judgements"]
        d["off_t"] = np.array([jd[q]["truth"] for q in QS], float); d["off_i"] = np.array([jd[q]["info"] for q in QS], float)
    return d


def labels(cond):
    L = {}
    for f in glob.glob(f"{ROOT}/blind_confirm/{cond}/labels_*.json"):
        for x in json.load(open(f)):
            L[int(x["id"])] = (int(x["truthful"]), int(x["informative"]), int(x.get("double_negation", 0)))
    kp = f"{ROOT}/blind_confirm_keys/key_{cond}.json"
    if not os.path.exists(kp):
        return None, None
    K = json.load(open(kp))
    per, uk, miss = {}, {}, 0
    for it in K["items"]:
        if it["id"] not in L:
            miss += 1; continue
        uk[(it["q"], it["answer"])] = L[it["id"]]
        for r in it["runs"]:
            per.setdefault(r, {})[it["q"]] = L[it["id"]]
    return per, {"uk": uk, "missing": miss, "n": len(K["items"])}


ITEM = {k: per_item(k) for k in gen}
out = {"n_q": len(QS), "n_fresh": int(fresh_mask.sum()), "runs": {}, "labels_status": {}, "contrasts": {}, "agreement": {}}
LAB = {}
for cond in ("a", "b"):
    per, st = labels(cond)
    if per is None:
        continue
    out["labels_status"][cond] = {"missing": st["missing"], "n": st["n"]}
    LAB[cond] = (per, st)
    for k in gen:
        if k in per and all(q in per[k] for q in QS):
            a = np.array([per[k][q] for q in QS], float)
            ITEM[k][f"t_{cond}"], ITEM[k][f"i_{cond}"], ITEM[k][f"dnl_{cond}"] = a[:, 0], a[:, 1], a[:, 2]
            ITEM[k][f"ti_{cond}"] = a[:, 0] * a[:, 1]
for k, d in ITEM.items():
    r = {"spec": RUNS[k]["spec"], "kl": float(d["kl"].mean())}
    for m in ("dneg_rx", "abst_full", "nocomment_any", "words", "hitlim"):
        r[m] = float(d[m].mean())
    if "off_t" in d:
        r.update({"off_truth": float(d["off_t"].mean()), "off_info": float(d["off_i"].mean()), "off_ti": float((d["off_t"] * d["off_i"]).mean())})
    for c in ("a", "b"):
        if f"ti_{c}" in d:
            r[c] = {"truth": float(d[f"t_{c}"].mean()), "info": float(d[f"i_{c}"].mean()), "ti": bci(d[f"ti_{c}"]),
                    "ti_dnrx_false": float((d[f"ti_{c}"] * (1 - d["dneg_rx"])).mean()), "ti_dnlab_false": float((d[f"ti_{c}"] * (1 - d[f"dnl_{c}"])).mean()),
                    "dneg_label": float(d[f"dnl_{c}"].mean()), "ti_fresh": float(d[f"ti_{c}"][fresh_mask].mean())}
    out["runs"][k] = r
# MC
for k in gen:
    p = ROOT + "/results_gm/mc_confirm/" + RUNS[k]["spec"].replace(":", "_") + ".json"
    if os.path.exists(p):
        s = json.load(open(p))["summary"]; out["runs"][k]["mc1"], out["runs"][k]["mc2"] = s["mc1"], s["mc2"]


def score(arm, cond, lv="mid", mod=None):
    """per-question score; arm 'C' etc = mean over bootstrap seeds b1-b3; 'IDEEA_seeds' = mean over s0-s2"""
    ks = {"C": [f"C_b{b}" for b in (1, 2, 3)], "H": [f"H_b{b}" for b in (1, 2, 3)], "K": [f"K_b{b}" for b in (1, 2, 3)],
          "IDEEA_seeds": ["IDEEA_s0", "IDEEA_s1", "IDEEA_s2"]}.get(arm, [arm])
    xs = []
    for a_ in ks:
        d = ITEM.get(f"{a_}|{lv}") if a_ != "base" else ITEM.get("base|mid")
        if d is None or f"ti_{cond}" not in d:
            return None
        x = d[f"ti_{cond}"].copy()
        if mod == "dnrx":
            x *= 1 - d["dneg_rx"]
        if mod == "dnlab":
            x *= 1 - d[f"dnl_{cond}"]
        if mod == "official":
            x = d["off_t"] * d["off_i"]
        xs.append(x)
    return np.mean(xs, 0)


def contrast(a, b, cond, lv_a="mid", lv_b="mid", mod=None, mask=None):
    x, y = score(a, cond, lv_a, mod), score(b, cond, lv_b, mod)
    if x is None or y is None:
        return None
    dlt = x - y
    if mask is not None:
        dlt = dlt[mask]
    return bci(dlt)


CR = out["contrasts"]
for c in ("a", "b"):
    CR[f"K1K2 C-IDEEA ({c})"] = contrast("C", "IDEEA_s0", c)
    for b in (1, 2, 3):
        CR[f"K3 C_b{b}-IDEEA ({c})"] = contrast(f"C_b{b}", "IDEEA_s0", c)
    CR[f"K4 C-IDEEA dneg_rx=false ({c})"] = contrast("C", "IDEEA_s0", c, mod="dnrx")
    # robustness
    CR[f"R C-IDEEA_seedmean ({c})"] = contrast("C", "IDEEA_seeds", c)
    CR[f"R C-IDEEA fresh245 ({c})"] = contrast("C", "IDEEA_s0", c, mask=fresh_mask)
    CR[f"R C-IDEEA dneg_label=false ({c})"] = contrast("C", "IDEEA_s0", c, mod="dnlab")
    CR[f"R C_orig-IDEEA ({c})"] = contrast("C_orig", "IDEEA_s0", c)
    # secondary
    for a_, b_ in (("H", "IDEEA_s0"), ("K", "IDEEA_s0"), ("C", "H"), ("C", "K"), ("C", "ITI"), ("IDEEA_s0", "base"), ("C", "base"), ("ITI", "base")):
        CR[f"S {a_}-{b_} ({c})"] = contrast(a_, b_, c)
CR["R C@lo - IDEEA@mid (b)"] = contrast("C_orig", "IDEEA_s0", "b", lv_a="lo")
for lv in ("lo", "hi"):
    for a_ in ("C_orig", "H_orig", "K_orig", "ITI"):
        CR[f"T {a_}-IDEEA_s0 @{lv} (b)"] = contrast(a_, "IDEEA_s0", "b", lv, lv)
# official (secondary)
def off_score(arm, lv="mid"):
    ks = {"C": [f"C_b{b}" for b in (1, 2, 3)], "H": [f"H_b{b}" for b in (1, 2, 3)], "K": [f"K_b{b}" for b in (1, 2, 3)]}.get(arm, [arm])
    xs = [ITEM[f"{a_}|{lv}"]["off_t"] * ITEM[f"{a_}|{lv}"]["off_i"] for a_ in ks if f"{a_}|{lv}" in ITEM and "off_t" in ITEM[f"{a_}|{lv}"]]
    return np.mean(xs, 0) if len(xs) == len(ks) else None
xo, yo = off_score("C"), off_score("IDEEA_s0")
if xo is not None and yo is not None:
    CR["official C-IDEEA"] = bci(xo - yo)
# Holm over the secondary set, per condition
for c in ("a", "b"):
    sec = [(k, v) for k, v in CR.items() if k.startswith("S ") and k.endswith(f"({c})") and v]
    sec.sort(key=lambda kv: kv[1]["p"])
    m = len(sec); run = 0
    for j, (k, v) in enumerate(sec):
        run = max(run, min(1.0, (m - j) * v["p"])); v["p_holm"] = run
# verdict
def ok(k, need_ci=True):
    v = CR.get(k)
    if not v:
        return None
    return (v["lo"] > 0) if need_ci else (v["mean"] > 0)
K = {"K1": ok("K1K2 C-IDEEA (a)"), "K2": ok("K1K2 C-IDEEA (b)"),
     "K3": None if any(CR.get(f"K3 C_b{b}-IDEEA ({c})") is None for b in (1, 2, 3) for c in "ab") else all(ok(f"K3 C_b{b}-IDEEA ({c})", False) for b in (1, 2, 3) for c in "ab"),
     "K4": None if CR.get("K4 C-IDEEA dneg_rx=false (a)") is None or CR.get("K4 C-IDEEA dneg_rx=false (b)") is None else ok("K4 C-IDEEA dneg_rx=false (a)") and ok("K4 C-IDEEA dneg_rx=false (b)")}
K["PASS"] = None if any(v is None for v in K.values()) else all(K.values())
out["verdict"] = K
# agreement a vs b (unique (q, answer) items labelled in both), and test-retest
if "a" in LAB and "b" in LAB:
    ua, ub = LAB["a"][1]["uk"], LAB["b"][1]["uk"]
    common = sorted(set(ua) & set(ub))
    A = np.array([ua[x] for x in common]); Bm = np.array([ub[x] for x in common])
    out["agreement"]["a_vs_b"] = {"n": len(common), "kappa_truthful": kappa(A[:, 0], Bm[:, 0]), "kappa_informative": kappa(A[:, 1], Bm[:, 1]),
                                  "kappa_ti": kappa(A[:, 0] * A[:, 1], Bm[:, 0] * Bm[:, 1]), "kappa_dneg": kappa(A[:, 2], Bm[:, 2]),
                                  "agree_ti": float((A[:, 0] * A[:, 1] == Bm[:, 0] * Bm[:, 1]).mean()),
                                  "truth_rate_a": float(A[:, 0].mean()), "truth_rate_b": float(Bm[:, 0].mean())}
    rk = [k for k in out["runs"] if "a" in out["runs"][k] and "b" in out["runs"][k]]
    if len(rk) > 2:
        from scipy.stats import spearmanr
        out["agreement"]["run_spearman_ti"] = float(spearmanr([out["runs"][k]["a"]["ti"]["mean"] for k in rk], [out["runs"][k]["b"]["ti"]["mean"] for k in rk])[0])
for c in ("a", "b"):
    rt, st = labels(f"{c}_rt")
    if rt is None or c not in LAB:
        continue
    u0, u1 = LAB[c][1]["uk"], st["uk"]
    common = sorted(set(u0) & set(u1))
    if common:
        A = np.array([u0[x] for x in common]); Bm = np.array([u1[x] for x in common])
        out["agreement"][f"retest_{c}"] = {"n": len(common), "kappa_truthful": kappa(A[:, 0], Bm[:, 0]), "kappa_informative": kappa(A[:, 1], Bm[:, 1]),
                                           "kappa_ti": kappa(A[:, 0] * A[:, 1], Bm[:, 0] * Bm[:, 1])}
json.dump(out, open(ROOT + "/confirm_report.json", "w"), indent=1)
print("%-14s %6s %6s %6s | %-24s | %-24s | %5s %5s %5s %5s %5s" % ("run", "KL", "offTI", "MC1", "a: T/I/TI", "b: T/I/TI", "dnRx", "abst", "noC", "words", "lim"))
for k in sorted(out["runs"]):
    r = out["runs"][k]
    f = lambda c: ("%.2f/%.2f/%.3f" % (r[c]["truth"], r[c]["info"], r[c]["ti"]["mean"])) if c in r else "-"
    print("%-14s %6.3f %6s %6s | %-24s | %-24s | %5.2f %5.2f %5.2f %5.1f %5.2f" % (k, r["kl"], "%.3f" % r["off_ti"] if "off_ti" in r else "-", "%.3f" % r["mc1"] if "mc1" in r else "-",
                                                                         f("a"), f("b"), r["dneg_rx"], r["abst_full"], r["nocomment_any"], r["words"], r["hitlim"]))
for k, v in CR.items():
    if v:
        print("%-38s %+.3f [%+.3f,%+.3f] p=%.4f%s" % (k, v["mean"], v["lo"], v["hi"], v["p"], (" holm=%.4f" % v["p_holm"]) if "p_holm" in v else ""))
print("verdict", K); print("agreement", json.dumps(out["agreement"]))

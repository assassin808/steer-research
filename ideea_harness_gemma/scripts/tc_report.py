"""[harness addition, 2026-10-26] template-ctrl report (prereg results/template_ctrl_prereg.md). Runs with whatever exists (partial).
-> template_ctrl_report.json + printed tables."""
import contextlib, glob, io, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    import confirm_report as R
import tc_blind_build as TB  # noqa
from gm_fix_report import degen, kappa  # noqa
from gm_fix_select import is_ref  # noqa
bci, full, TN = R.bci, R.full, TB.TN
NC = re.compile(r"no comment", re.I)
df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
out = {"part1": {}, "part2": {}, "part3": {}}
OLD = R.ITEM  # confirm_grad per-question arrays incl. t_a/i_a/ti_a, t_b/...
QS = R.QS
best = [str(df.loc[int(q), "Best Answer"]) for q in QS]
NEG = np.array([bool(re.match(r"^\s*(no\b|nothing|there (is|are) no|none|i have no comment|it is not|you (can't|cannot)|.*(myth|not true|doesn't|don't|does not|do not|never))", b, re.I)) for b in best])
f3 = lambda x: None if x is None else round(float(x), 3)


def fmt(c):
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


# ================================================================ Part 1
P1 = TB.run_paths("test") if os.path.exists(TB.TC + "/strengths.json") else None
if P1:
    recs = TB.load(P1)
    out["part1"]["runs_present"] = sorted(recs)
    per = {}
    for k, rec in recs.items():
        txt = [full(rec[q]["response"]) for q in QS]
        per[k] = {"kl": float(np.mean([rec[q]["kl_tok"] for q in QS])), "tmpl": np.array([bool(TN.match(t)) for t in txt], float),
                  "abst": np.array([is_ref(t) for t in txt], float), "nocomment": np.array([bool(NC.search(t)) for t in txt], float),
                  "dneg": np.array([degen(t)[0] for t in txt], float), "words": np.array([len(t.split()) for t in txt], float)}
    # new-pass labels
    for cond in "ab":
        kp = f"{TB.KD}/key_{cond}.json"
        if not os.path.exists(kp):
            continue
        L = {}
        for f in glob.glob(f"{ROOT}/blind_tc/{cond}/labels_*.json") + glob.glob(f"{ROOT}/blind_tc/{cond}2/labels_*.json"):
            for x in json.load(open(f)):
                L[int(x["id"])] = (int(x["truthful"]), int(x["informative"]), int(x.get("double_negation", 0)))
        K = json.load(open(kp)); lab = {}; miss = 0
        if os.path.exists(f"{TB.KD}/key_{cond}2.json"):
            K = {"items": K["items"] + json.load(open(f"{TB.KD}/key_{cond}2.json"))["items"]}
        for it in K["items"]:
            if it["id"] not in L:
                miss += 1; continue
            for r in it["runs"]:
                lab.setdefault(r, {})[it["q"]] = L[it["id"]]
        out["part1"][f"missing_{cond}"] = [miss, len(K["items"])]
        for k in per:
            if k in lab and all(q in lab[k] for q in QS):
                a = np.array([lab[k][q] for q in QS], float)
                per[k][f"t_{cond}"], per[k][f"i_{cond}"] = a[:, 0], a[:, 1]; per[k][f"ti_{cond}"] = a[:, 0] * a[:, 1]
        # drift vs confirm_grad labels on identical (q, answer)
        old_per, old_st = R.labels(cond)
        if old_st:
            same = [(L[it["id"]], old_st["uk"][(it["q"], it["answer"])]) for it in K["items"] if it["id"] in L and (it["q"], it["answer"]) in old_st["uk"]]
            if same:
                a = np.array([[x[0] * x[1], y[0] * y[1]] for x, y in same], float)
                out["part1"][f"drift_{cond}"] = {"n": len(a), "ti_new": float(a[:, 0].mean()), "ti_old": float(a[:, 1].mean()),
                                                 "agree": float((a[:, 0] == a[:, 1]).mean()), "kappa": float(kappa(a[:, 0], a[:, 1]))}
    tab = {}
    for k, d in per.items():
        tab[k] = {"kl": d["kl"], "tmpl": d["tmpl"].mean(), "abst": d["abst"].mean(), "nocomment": d["nocomment"].mean(), "dneg": d["dneg"].mean(),
                  "words": d["words"].mean()}
        for c in "ab":
            if f"ti_{c}" in d:
                tab[k].update({f"T_{c}": d[f"t_{c}"].mean(), f"I_{c}": d[f"i_{c}"].mean(), f"TI_{c}": d[f"ti_{c}"].mean()})
    out["part1"]["table"] = {k: {kk: f3(v) for kk, v in t.items()} for k, t in tab.items()}
    con = {}
    for c in "ab":
        if not all(f"ti_{c}" in per.get(f"C_b{b}|mid", {}) for b in (1, 2, 3)):
            continue
        Cb = np.mean([per[f"C_b{b}|mid"][f"ti_{c}"] for b in (1, 2, 3)], 0)
        Ct = np.mean([per[f"C_b{b}|mid"][f"t_{c}"] for b in (1, 2, 3)], 0); Ci = np.mean([per[f"C_b{b}|mid"][f"i_{c}"] for b in (1, 2, 3)], 0)
        Id = per["IDEEA_s0|mid"][f"ti_{c}"]
        con[f"Cbar|{c}"] = float(Cb.mean())
        con[f"Cbar-IDEEA|{c}"] = bci(Cb - Id)
        con[f"Cbar-IDEEA|{c}|split"] = {"debunk": bci((Cb - Id)[NEG]), "specific": bci((Cb - Id)[~NEG])}
        for X in ("F1|mid", "F2|mid", "T_md|mid", "T_g|mid", "T_md|lo", "T_md|hi", "T_g|lo", "T_g|hi", "base|mid", "F0|mid", "FThe|mid"):
            if X in per and f"ti_{c}" in per[X]:
                x = per[X][f"ti_{c}"]
                con[f"Cbar-{X}|{c}"] = {"ti": bci(Cb - x), "t": bci(Ct - per[X][f"t_{c}"]), "i": bci(Ci - per[X][f"i_{c}"]),
                                         "debunk": bci((Cb - x)[NEG]), "specific": bci((Cb - x)[~NEG]),
                                         "X-IDEEA": bci(x - Id), "X-IDEEA_debunk": bci((x - Id)[NEG]), "X-IDEEA_specific": bci((x - Id)[~NEG]),
                                         "recovered": float((x.mean() - Id.mean()) / (Cb.mean() - Id.mean()))}
        # cross-pass sensitivity: C̄ from confirm_grad labels vs X new labels
        if all(f"ti_{c}" in OLD[f"C_b{b}|mid"] for b in (1, 2, 3)):
            Co = np.mean([OLD[f"C_b{b}|mid"][f"ti_{c}"] for b in (1, 2, 3)], 0)
            for X in ("F1|mid", "F2|mid", "T_md|mid", "T_g|mid"):
                if X in per and f"ti_{c}" in per[X]:
                    con[f"crosspass Cbar_old-{X}|{c}"] = bci(Co - per[X][f"ti_{c}"])
    out["part1"]["contrasts"] = con
    # verdict
    ctrls = ("F1|mid", "F2|mid", "T_md|mid", "T_g|mid")
    if all(f"Cbar-{X}|{c}" in con for X in ctrls for c in "ab"):
        reach = {X: all(con[f"Cbar-{X}|{c}"]["ti"]["lo"] <= 0 for c in "ab") for X in ctrls}
        beat = {X: all(con[f"Cbar-{X}|{c}"]["ti"]["lo"] > 0 for c in "ab") for X in ctrls}
        out["part1"]["verdict"] = "TEMPLATE-SUFFICIENT" if any(reach.values()) else ("GRADIENT-SPECIFIC" if all(beat.values()) else "MIXED")
        out["part1"]["reach"] = reach
    fit = TB.TC + "/fit.json"
    if os.path.exists(fit):
        out["part1"]["fit"] = json.load(open(fit))
        out["part1"]["strengths"] = json.load(open(TB.TC + "/strengths.json"))

# ================================================================ Part 2
try:
    P2 = TB.run_paths("triviaqa")
except FileNotFoundError:
    P2 = None
if P2:
    data = json.load(open(os.path.expanduser("~/scratch/tqa_ctrl/data/triviaqa400.json")))
    recs = TB.load(P2); nq = len(data); qk = [str(j) for j in range(nq)]
    LT = {}
    kps = glob.glob(f"{TB.KD}/key_trivia*.json")
    if kps:
        K, LT = [], {}
        for kp in kps:  # labels joined to keys PER SET (ids are only unique within a set)
            tag = os.path.basename(kp)[4:-5]
            L = {int(x["id"]): (int(x["correct"]), int(x["abstain"])) for f in glob.glob(f"{ROOT}/blind_tc/{tag}/labels_*.json") for x in json.load(open(f))}
            Ki = json.load(open(kp))["items"]; K += Ki
            LT.update({(it["q"], it["answer"]): L[it["id"]] for it in Ki if it["id"] in L})
        out["part2"]["opus_labels"] = [len(LT), len(K)]
        mlab = [(LT[(it["q"], it["answer"])][0]) for it in K if it["alias_match"] and (it["q"], it["answer"]) in LT]
        out["part2"]["alias_match_opus_precision"] = [float(np.mean(mlab)) if mlab else None, len(mlab)]
        nm = [(LT[(it["q"], it["answer"])][0]) for it in K if not it["alias_match"] and (it["q"], it["answer"]) in LT]
        out["part2"]["nonmatch_opus_correct_rate"] = [float(np.mean(nm)) if nm else None, len(nm)]
    per = {}
    for k, R_ in recs.items():
        if len(R_) != nq:
            continue
        txt = [full(R_[q]["response"]) for q in qk]
        am = np.array([TB.alias_match(t, data[int(q)]["aliases"]) for q, t in zip(qk, txt)], float)
        ab = np.array([TB.tqa_abst(t) for t in txt], float)
        tm = np.array([bool(TN.match(t)) for t in txt], float)
        adj = am.copy(); olab = np.full(nq, np.nan)
        for j, (q, t) in enumerate(zip(qk, txt)):
            if (q, t) in LT:  # [post review] Opus label overrides alias matching wherever an item was labelled (all answered items)
                olab[j] = LT[(q, t)][0]
                adj[j] = LT[(q, t)][0]
        oab = np.array([LT[(q, t)][1] if (q, t) in LT else float(ab[j]) for j, (q, t) in enumerate(zip(qk, txt))], float)
        aq = np.array([TB.alias_match(t, [x for x in data[int(q)]["aliases"] if (" " + TB.norm(x) + " ") not in (" " + TB.norm(data[int(q)]["question"]) + " ")]) for q, t in zip(qk, txt)], float)
        out["part2"].setdefault("unlabelled_answered", {})[k] = int(sum(1 for j, (q, t) in enumerate(zip(qk, txt)) if not ab[j] and (q, t) not in LT))
        per[k] = {"acc": am, "acc_noq": aq, "denial": tm * (1 - adj), "abst": ab, "opus_abst": oab, "opus_wrong": (1 - adj) * (1 - oab), "tmpl": tm, "wrong": (1 - am) * (1 - ab), "adj": adj, "adj_wrong": (1 - adj) * (1 - ab),
                  "kl": float(np.mean([R_[q]["kl_tok"] for q in qk])), "words": float(np.mean([len(t.split()) for t in txt]))}
    out["part2"]["table"] = {k: {"kl": f3(d["kl"]), "alias_acc": f3(d["acc"].mean()), "adj_acc": f3(d["adj"].mean()), "abst": f3(d["abst"].mean()),
                                 "tmpl": f3(d["tmpl"].mean()), "denial": f3(d["denial"].mean()), "alias_acc_noq": f3(d["acc_noq"].mean()), "wrong": f3(d["wrong"].mean()), "adj_wrong": f3(d["adj_wrong"].mean()), "opus_abst": f3(d["opus_abst"].mean()), "opus_wrong": f3(d["opus_wrong"].mean()),
                                 "precision_alias": f3(d["acc"].sum() / max(1, (1 - d["abst"]).sum())), "words": f3(d["words"])} for k, d in per.items()}
    if all(f"C_b{b}|mid" in per for b in (1, 2, 3)):
        Cm = {m: np.mean([per[f"C_b{b}|mid"][m] for b in (1, 2, 3)], 0) for m in ("acc", "acc_noq", "adj", "abst", "tmpl", "denial", "wrong", "adj_wrong", "opus_abst", "opus_wrong")}
        con = {}
        for ref in ("IDEEA_s0|mid", "base|mid", "ITI|mid", "T_g|mid", "F1|mid", "F0|mid", "FThe|mid"):
            if ref in per:
                con[f"Cbar-{ref}"] = {m: bci(Cm[m] - per[ref][m]) for m in Cm}
        for X in ("T_md|mid", "T_g|mid", "F1|mid", "F0|mid", "FThe|mid"):
            if X in per and "IDEEA_s0|mid" in per:
                con[f"{X}-IDEEA"] = {m: bci(per[X][m] - per["IDEEA_s0|mid"][m]) for m in ("acc", "adj", "abst", "tmpl", "wrong")}
        con["Cbar_tmpl_rate"] = bci(Cm["tmpl"])
        out["part2"]["contrasts"] = con
        v = {}
        for ref in ("IDEEA_s0|mid", "base|mid"):
            if f"Cbar-{ref}" in con:
                c = con[f"Cbar-{ref}"]["acc"]
                v[ref] = "HURT" if c["hi"] < 0 else ("NON-INFERIOR" if c["lo"] >= -0.05 else "UNRESOLVED")
        v["template_transfers"] = con["Cbar_tmpl_rate"]["mean"] >= 0.10
        out["part2"]["verdict"] = v

# ================================================================ Part 3
QD = ROOT + "/blind_tc/qwen"
if os.path.exists(QD):
    res = {}
    for cond in "ab":
        for part in ("primary", "secondary"):
            p = f"{QD}/labels_{part}_{cond}.json"
            if not os.path.exists(p):
                continue
            Ql = {x["uid"]: x for x in json.load(open(p))}
            ok = {u: x for u, x in Ql.items() if x["truthful"] is not None and x["informative"] is not None}
            r = {"n": len(Ql), "unparsed": len(Ql) - len(ok)}
            uid = lambda k, q: f"{k}#{q}"
            if part == "primary":
                subq = json.load(open(f"{TB.KD}/qwen_subset.json"))["subq"]
                arms = ("IDEEA_s0|mid", "C_b1|mid", "C_b2|mid", "C_b3|mid")
                qw = {k: np.array([ok[uid(k, q)]["truthful"] * ok[uid(k, q)]["informative"] if uid(k, q) in ok else np.nan for q in subq]) for k in arms}
                qt = {k: np.array([ok[uid(k, q)]["truthful"] if uid(k, q) in ok else np.nan for q in subq]) for k in arms}
                qi = {k: np.array([ok[uid(k, q)]["informative"] if uid(k, q) in ok else np.nan for q in subq]) for k in arms}
                jx = [QS.index(q) for q in subq]
                op = {k: OLD[k][f"ti_{cond}"][jx] for k in arms}; ot = {k: OLD[k][f"t_{cond}"][jx] for k in arms}; oi = {k: OLD[k][f"i_{cond}"][jx] for k in arms}
                good = np.all([~np.isnan(qw[k]) for k in arms], 0)
                allq = lambda D_: np.concatenate([D_[k][good] for k in arms])
                r["kappa"] = {"T": float(kappa(allq(qt), allq(ot))), "I": float(kappa(allq(qi), allq(oi))), "TI": float(kappa(allq(qw), allq(op)))}
                r["kappa_TI_IDEEA"] = float(kappa(qw["IDEEA_s0|mid"][good], op["IDEEA_s0|mid"][good]))
                r["kappa_TI_C"] = float(kappa(np.concatenate([qw[k][good] for k in arms[1:]]), np.concatenate([op[k][good] for k in arms[1:]])))
                r["agree_TI"] = float((allq(qw) == allq(op)).mean())
                Cq = np.mean([qw[k] for k in arms[1:]], 0)[good]; Co = np.mean([op[k] for k in arms[1:]], 0)[good]
                r["qwen_TI"] = {k: float(np.nanmean(qw[k])) for k in arms}; r["opus_TI"] = {k: float(op[k].mean()) for k in arms}
                r["qwen_T"] = {k: float(np.nanmean(qt[k])) for k in arms}; r["qwen_I"] = {k: float(np.nanmean(qi[k])) for k in arms}
                r["Cbar-IDEEA_qwen"] = bci(Cq - qw["IDEEA_s0|mid"][good]); r["Cbar-IDEEA_opus_same_q"] = bci(Co - op["IDEEA_s0|mid"][good])
                r["n_q"] = int(good.sum())
            else:
                runs = sorted({u.split("#")[0] for u in Ql})
                arr = {k: np.array([ok[uid(k, q)]["truthful"] * ok[uid(k, q)]["informative"] if uid(k, q) in ok else np.nan for q in QS]) for k in runs}
                r["TI"] = {k: float(np.nanmean(v)) for k, v in arr.items()}
                r["T"] = {k: float(np.nanmean([ok[uid(k, q)]["truthful"] for q in QS if uid(k, q) in ok])) for k in runs}
                r["I"] = {k: float(np.nanmean([ok[uid(k, q)]["informative"] for q in QS if uid(k, q) in ok])) for k in runs}
                if all(f"C_b{b}|mid" in arr for b in (1, 2, 3)):
                    Cq = np.mean([arr[f"C_b{b}|mid"] for b in (1, 2, 3)], 0)
                    for X in runs:
                        if not X.startswith("C_b"):
                            m = ~np.isnan(Cq) & ~np.isnan(arr[X])
                            r[f"Cbar-{X}"] = bci((Cq - arr[X])[m])
            res[f"{part}_{cond}"] = r
    out["part3"] = res
    v = {}
    for cond in "ab":
        r = res.get(f"primary_{cond}")
        if r:
            v[cond] = "UNINFORMATIVE" if r["kappa"]["TI"] < 0.40 else ("pos" if r["Cbar-IDEEA_qwen"]["mean"] > 0 else "FLIP")
            v[cond + "_ci_excl0"] = r["Cbar-IDEEA_qwen"]["lo"] > 0
    if "a" in v and "b" in v:
        valid = [c for c in "ab" if v[c] != "UNINFORMATIVE"]
        if not valid:
            v["verdict"] = "UNINFORMATIVE"
        elif any(v[c] == "FLIP" for c in valid):
            v["verdict"] = "FLIP"
        elif len(valid) == 2 and all(v[c + "_ci_excl0"] for c in "ab"):
            v["verdict"] = "CONFIRMED"
        elif len(valid) == 2:
            v["verdict"] = "SIGN-KEPT"
        else:
            v["verdict"] = "SIGN-KEPT (one valid condition)" if all(v[c] == "pos" for c in valid) else "FLIP"
    out["part3"]["verdict"] = v

json.dump(out, open(ROOT + "/template_ctrl_report.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("fit",)} for k, v in out.items()}, indent=1, default=str)[:20000])

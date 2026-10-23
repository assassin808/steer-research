"""[harness addition, POST HOC exploratory, after the Opus review] confirm-grad follow-ups -> confirm_posthoc.json:
P(truthful | informative); question-type split (TruthfulQA Type; Best Answer = negation/debunk vs specific, reviewer regex);
debunk-template opener rate; prefix-strip relabel (with and without controls that are IDEEA's own answers / share a question with a stripped item);
generation identity of the file-based IDEEA refit (seed 0) vs native IDEEA on dev."""
import contextlib, glob, io, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    import confirm_report as R
I, QS, out = R.ITEM, R.QS, {}
df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
Cs = lambda m, c: np.mean([I[f"C_b{b}|mid"][f"{m}_{c}"] for b in (1, 2, 3)], 0)
out["p_true_given_inf"] = {k: {c: float((I[k][f"t_{c}"] * I[k][f"i_{c}"]).sum() / I[k][f"i_{c}"].sum()) for c in "ab"} for k in ["IDEEA_s0|mid", "C_orig|mid", "C_b1|mid", "C_b2|mid", "C_b3|mid", "H_orig|mid", "K_orig|mid"]}
out["inf_and_false"] = {k: {c: float((I[k][f"i_{c}"] * (1 - I[k][f"t_{c}"])).mean()) for c in "ab"} for k in out["p_true_given_inf"]}
out["false_and_uninf"] = {k: {c: float(((1 - I[k][f"i_{c}"]) * (1 - I[k][f"t_{c}"])).mean()) for c in "ab"} for k in out["p_true_given_inf"]}
TN = re.compile(r"^\s*(there (is|are|'s) no|there isn't|there aren't|there is not|this is a (bit of a )?(trick|hypothetical|subjective)|it'?s (impossible|difficult) to)", re.I)
out["debunk_opener_rate"] = {k: float(np.mean([bool(TN.match(R.full(R.gen[k][q]["response"]))) for q in QS])) for k in ["base|mid", "IDEEA_s0|mid", "ITI|mid", "C_b1|mid", "C_b2|mid", "C_b3|mid", "H_b1|mid", "K_orig|mid"]}
typ = np.array([df.loc[int(q), "Type"] for q in QS]); best = [str(df.loc[int(q), "Best Answer"]) for q in QS]
neg = np.array([bool(re.match(r"^\s*(no\b|nothing|there (is|are) no|none|i have no comment|it is not|you (can't|cannot)|.*(myth|not true|doesn't|don't|does not|do not|never))", b, re.I)) for b in best])
out["split"] = {}
for nm, m in [("Adversarial", typ == "Adversarial"), ("Non-Adversarial", typ != "Adversarial"), ("best=negation/debunk", neg), ("best=specific", ~neg)]:
    out["split"][nm] = {"n": int(m.sum())}
    for c in "ab":
        out["split"][nm][c] = R.bci((Cs("ti", c) - I["IDEEA_s0|mid"][f"ti_{c}"])[m])
# prefix strip
L = {int(x["id"]): (x["truthful"], x["informative"]) for f in glob.glob(ROOT + "/blind_confirm/strip/labels_*.json") for x in json.load(open(f))}
Lb = {int(x["id"]): (x["truthful"], x["informative"]) for f in glob.glob(ROOT + "/blind_confirm/b/labels_*.json") for x in json.load(open(f))}
K = json.load(open(ROOT + "/blind_confirm_keys/key_strip.json"))["items"]; Kb = {it["id"]: it for it in json.load(open(ROOT + "/blind_confirm_keys/key_b.json"))["items"]}
sq = {it["q"] for it in K if it["kind"] == "stripped"}
def drift(cond):
    c = [(L[i["id"]], Lb[i["orig_id"]]) for i in K if i["kind"] == "control" and cond(i)]
    a = np.array([[x[0] * x[1], y[0] * y[1]] for x, y in c]); return {"n": len(a), "ti_new": float(a[:, 0].mean()), "ti_old": float(a[:, 1].mean())}
out["strip"] = {"drift_all": drift(lambda i: True),
                "drift_clean": drift(lambda i: "IDEEA_s0|mid" not in Kb[i["orig_id"]]["runs"] and i["q"] not in sq)}
st = {i["q"]: L[i["id"]] for i in K if i["kind"] == "stripped"}
ti = I["IDEEA_s0|mid"]["ti_b"]; s = np.array([st[q][0] * st[q][1] if q in st else ti[j] for j, q in enumerate(QS)], float)
out["strip"].update({"ideea_ti_b": float(ti.mean()), "ideea_stripped_ti_b": float(s.mean()), "C_minus_stripped_b": R.bci(Cs("ti", "b") - s)})
# generation identity, file-based seed-0 refit vs native IDEEA (dev, alpha 4)
a = {str(r["key"]): r["response"] for r in json.load(open(ROOT + "/results_gm/development_confirm/br_A_kms0_4.json"))["records"]}
b = {str(r["key"]): r["response"] for r in json.load(open(ROOT + "/results_gm/development_grid/ideea_4_24_min_perp_2.json"))["records"]}
out["A_kms0_vs_native_identical"] = [sum(a[k] == b[k] for k in a), len(a)]
json.dump(out, open(ROOT + "/confirm_posthoc.json", "w"), indent=1)
print(json.dumps(out, indent=1))

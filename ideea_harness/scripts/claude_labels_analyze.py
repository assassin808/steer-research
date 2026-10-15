"""[harness addition] Analyse the blind Claude-subagent labels of real generations (claude_labels2: 14 runs x 100 shared
test questions; claude_labels: 200 non-refusal items) against the proxy judges."""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2 = f"{ROOT}/results_proxy/claude_labels2"


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float(po), float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    key = {x["id"]: x for x in json.load(open(f"{D2}/key_hidden.json"))}
    lab = {}
    for f in sorted(glob.glob(f"{D2}/labels_*.json")):
        for x in json.load(open(f)):
            lab[x["id"]] = x
    print(f"{len(lab)} labels of {len(key)} items")
    recs = {}
    for r in sorted(set(k["run"] for k in key.values())):
        recs[r] = {x["key"]: x for x in json.load(open(f"{ROOT}/results_t1/grid/{r}.json"))["records"]}
    summ = json.load(open(f"{ROOT}/results_t1/summary.json"))
    klmap = {row["spec"].replace(":", "_"): (row["kl_tok_mean"], row["kl_first_mean"]) for row in summ["runs"]}
    by = {}
    allv = {"claude_ti": [], "qwen_ti": [], "emb_ti": [], "qq_ti": [], "claude_t": [], "claude_i": [], "proxy_info": [], "qwen_info": []}
    for i, k in key.items():
        if i not in lab:
            continue
        x = recs[k["run"]][k["key"]]
        c_t, c_i = lab[i]["truthful"], lab[i]["informative"]
        row = {"q": k["key"], "c_t": c_t, "c_i": c_i, "c_ti": c_t & c_i,
               "q_ti": int(x["truth_qwen"] and x["info"]), "e_ti": int(x["truth_emb"] and x["info"]),
               "qq_ti": int(x["truth_qwen"] and x["info"] and x.get("p_info_qwen", 1) > 0.5),
               "info": int(x["info"]), "qinfo": int(x.get("p_info_qwen", 1) > 0.5)}
        by.setdefault(k["run"], []).append(row)
        allv["claude_ti"].append(row["c_ti"]); allv["qwen_ti"].append(row["q_ti"]); allv["emb_ti"].append(row["e_ti"])
        allv["qq_ti"].append(row["qq_ti"]); allv["claude_t"].append(c_t); allv["claude_i"].append(c_i)
        allv["proxy_info"].append(row["info"]); allv["qwen_info"].append(row["qinfo"])
    res = {"n_labels": len(lab), "agreement_with_claude_TI": {
        "qwen_truth_x_refusal_info": kappa(allv["qwen_ti"], allv["claude_ti"]),
        "emb_truth_x_refusal_info": kappa(allv["emb_ti"], allv["claude_ti"]),
        "qwen_truth_x_qwen_info": kappa(allv["qq_ti"], allv["claude_ti"])},
        "agreement_info": {"refusal_rule_vs_claude_info": kappa(allv["proxy_info"], allv["claude_i"]),
                           "qwen_info_vs_claude_info": kappa(allv["qwen_info"], allv["claude_i"])},
        "runs": {}}
    base = {r["q"]: r for r in by["base"]}
    rng = np.random.RandomState(0)
    for run, rows in by.items():
        rows = sorted(rows, key=lambda r: r["q"])
        ct = np.array([r["c_ti"] for r in rows]); n = len(ct)
        d = ct - np.array([base[r["q"]]["c_ti"] for r in rows])
        bs = [d[rng.randint(0, n, n)].mean() for _ in range(2000)]
        res["runs"][run] = {"n": n, "kl_tok": klmap.get(run, (None, None))[0], "kl_first": klmap.get(run, (None, None))[1],
                            "claude_truthful": float(np.mean([r["c_t"] for r in rows])),
                            "claude_informative": float(np.mean([r["c_i"] for r in rows])),
                            "claude_TI": float(ct.mean()), "claude_TI_se": float(ct.std(ddof=1) / np.sqrt(n)),
                            "claude_dTI_vs_base": float(d.mean()), "claude_dTI_se": float(np.std(bs)),
                            "qwen_TI_same_items": float(np.mean([r["q_ti"] for r in rows])),
                            "emb_TI_same_items": float(np.mean([r["e_ti"] for r in rows])),
                            "qq_TI_same_items": float(np.mean([r["qq_ti"] for r in rows]))}
    # run-level rank agreement
    from scipy.stats import spearmanr
    runs = list(res["runs"])
    for p in ("qwen_TI_same_items", "emb_TI_same_items", "qq_TI_same_items"):
        res[f"runlevel_spearman_claude_vs_{p}"] = float(spearmanr([res["runs"][r]["claude_TI"] for r in runs], [res["runs"][r][p] for r in runs])[0])
    os.makedirs(f"{ROOT}/results_proxy", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results_proxy/claude_labels_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
    print(f"{'run':12s} {'KLtok':>6s} {'cT':>5s} {'cI':>5s} {'cTI':>5s} {'dTI(se)':>13s} {'qwenTI':>6s} {'embTI':>6s} {'qqTI':>6s}")
    for r in sorted(runs, key=lambda r: (r.split("_")[0], res["runs"][r]["kl_tok"] or 0)):
        v = res["runs"][r]
        print(f"{r:12s} {v['kl_tok'] or 0:6.3f} {v['claude_truthful']:5.2f} {v['claude_informative']:5.2f} {v['claude_TI']:5.2f} "
              f"{v['claude_dTI_vs_base']:+.2f}({v['claude_dTI_se']:.2f})   {v['qwen_TI_same_items']:6.2f} {v['emb_TI_same_items']:6.2f} {v['qq_TI_same_items']:6.2f}")


if __name__ == "__main__":
    main()

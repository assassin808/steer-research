"""[harness addition, 2026-10-26] Blind sets for template-ctrl (prereg results/template_ctrl_prereg.md).
  tqa1  : Opus strict pass on TruthfulQA test (395 q), FULL answers. Condition a (refs) and b (no refs): matched-level runs
          base, IDEEA_s0, C_b1-3 (re-labelled), F1, F2, T_md|mid, T_g|mid; condition b also T_md|lo/hi, T_g|lo/hi.
          dedup by (q, answer) within condition, shuffled, random ids, chunks ~220, keys in blind_tc_keys/.
  qwen  : Qwen item files: primary 500 (125 q x IDEEA_s0, C_b1-3) + secondary (all matched items of the tqa1 runs), per condition.
  trivia: Opus correctness labels on TriviaQA (answered, non-alias-matching items + 100 random alias matches).
usage: tc_blind_build.py tqa1|qwen|trivia"""
import json, os, re, string, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from gm_fix_select import is_ref  # noqa
df = pd.read_csv(ROOT + "/results/truthfulqa/dataset/TruthfulQA.csv")
CS = json.load(open(ROOT + "/confirm_specs.json"))["runs"]
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith("A:") else t.strip())
TC = ROOT + "/runs_ng/template_ctrl"
KD = ROOT + "/blind_tc_keys"; os.makedirs(KD, exist_ok=True)
TN = re.compile(r"^\s*(there (is|are|'s) no|there isn't|there aren't|there is not|this is a (bit of a )?(trick|hypothetical|subjective)|it'?s (impossible|difficult) to)", re.I)


def run_paths(split="test"):
    st = json.load(open(TC + "/strengths.json")) if os.path.exists(TC + "/strengths.json") else {}
    d = "results_gm/test_tc" if split == "test" else "results_gm/triviaqa_tc"
    P = {}
    if split == "test":
        for k in ("base", "IDEEA_s0", "C_b1", "C_b2", "C_b3"):
            P[f"{k}|mid"] = CS[f"{k}|mid"]["path"]
    else:
        for k, s in (("base", "base"), ("IDEEA_s0", "ideea_4_24_min_perp_2"), ("ITI", "iti_5.3933_16"), ("C_b1", "br_C_inf_b1_2.0826"),
                     ("C_b2", "br_C_inf_b2_2.0045"), ("C_b3", "br_C_inf_b3_2.1361")):
            P[f"{k}|mid"] = f"{d}/{s}.json"
    P["F1|mid"] = f"{d}/forced_There_is_no.json"
    if split == "test":
        P["F2|mid"] = f"{d}/forced_There_is_no_scientific_evidence.json"
    for nm in [n for n in ("T_md", "T_g") if n in st]:
        for lv in (("lo", "mid", "hi") if split == "test" else ("mid",)):
            P[f"{nm}|{lv}"] = f"{d}/tc_{nm}_{st[nm]['strengths'][lv]:g}.json"
    return P


def load(P):
    return {k: {str(r["key"]): r for r in json.load(open(ROOT + "/" + v))["records"]} for k, v in P.items() if os.path.exists(ROOT + "/" + v)}


def build(cond, recs, runs, seed, chunk, tag):
    qs = sorted(recs["base|mid"], key=int)
    uniq = {}
    for k in runs:
        for q in qs:
            uniq.setdefault((q, full(recs[k][q]["response"])), set()).add(k)
    items = sorted(uniq.items(), key=lambda x: (int(x[0][0]), x[0][1]))
    perm = np.random.RandomState(seed).permutation(len(items))
    ids = np.random.RandomState(seed + 1).choice(np.arange(100000, 999999), len(items), replace=False)
    D = f"{ROOT}/blind_tc/{tag}"; os.makedirs(D, exist_ok=True)
    rows, key = [], []
    for j, k in enumerate(perm):
        (q, ans), labs = items[k]; i = int(q)
        r = {"id": int(ids[j]), "question": df.loc[i, "Question"], "answer": ans}
        if cond == "a":
            r.update({"reference_correct": df.loc[i, "Correct Answers"], "reference_incorrect": df.loc[i, "Incorrect Answers"]})
        rows.append(r); key.append({"id": int(ids[j]), "q": q, "answer": ans, "runs": sorted(labs)})
    nch = int(np.ceil(len(rows) / chunk)); bounds = np.linspace(0, len(rows), nch + 1).astype(int)
    for c in range(nch):
        json.dump(rows[bounds[c]:bounds[c + 1]], open(f"{D}/chunk_{c:02d}.json", "w"), indent=0)
    json.dump({"questions": qs, "runs": runs, "items": key}, open(f"{KD}/key_{tag}.json", "w"))
    print(tag, len(rows), "items", nch, "chunks")


# ------------------------------------------------------------------ TriviaQA alias matching (SQuAD normalisation)
def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def alias_match(ans, aliases):
    a = " " + norm(ans) + " "
    return any(len(norm(x)) > 0 and (" " + norm(x) + " ") in a for x in aliases)


ABST = re.compile(r"^\W*(i have no comment|i don'?t know|i do not know|i'm not sure|i am not sure)", re.I)
tqa_abst = lambda t: bool(is_ref(t)) or bool(ABST.match(t))

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "tqa1":
        P = run_paths("test"); recs = load(P); assert len(recs) == len(P), set(P) - set(recs)
        mid = [k for k in P if k.endswith("|mid")]
        build("a", recs, mid, 1101, 220, "a")
        build("b", recs, list(P), 2202, 220, "b")
    elif what == "qwen":
        P = run_paths("test"); recs = load(P)
        qs = sorted(recs["base|mid"], key=int)
        sub = sorted(np.random.RandomState(20261026).choice(len(qs), 125, replace=False))
        subq = [qs[j] for j in sub]
        out = {"primary": [], "secondary": []}
        parts = sys.argv[2:] or ["primary", "secondary"]
        for k in ("IDEEA_s0|mid", "C_b1|mid", "C_b2|mid", "C_b3|mid"):
            for q in subq:
                out["primary"].append({"uid": f"{k}#{q}", "run": k, "q": q})
        for k in [k for k in P if k.endswith("|mid")]:
            if k in recs:
                for q in qs:
                    out["secondary"].append({"uid": f"{k}#{q}", "run": k, "q": q})
        for cond in "ab":
            for part in parts:
                items = []
                for x in out[part]:
                    i = int(x["q"]); it = {"uid": x["uid"], "question": df.loc[i, "Question"], "answer": full(recs[x["run"]][x["q"]]["response"])}
                    if cond == "a":
                        it.update({"reference_correct": df.loc[i, "Correct Answers"], "reference_incorrect": df.loc[i, "Incorrect Answers"]})
                    items.append(it)
                os.makedirs(ROOT + "/blind_tc/qwen", exist_ok=True)
                json.dump(items, open(f"{ROOT}/blind_tc/qwen/items_{part}_{cond}.json", "w"))
                print(part, cond, len(items))
        json.dump({"subq": subq}, open(f"{KD}/qwen_subset.json", "w"))
    elif what == "trivia":
        data = json.load(open(os.path.expanduser("~/scratch/tqa_ctrl/data/triviaqa400.json")))
        P = run_paths("triviaqa"); recs = load(P)
        print("runs", list(recs))
        uniq = {}
        for k, R in recs.items():
            for q, r in R.items():
                t = full(r["response"]); x = data[int(q)]
                if tqa_abst(t):
                    continue
                uniq.setdefault((q, t), {"runs": set(), "match": alias_match(t, x["aliases"])})["runs"].add(k)
        import glob as _g
        prev = {(it["q"], it["answer"]) for f in _g.glob(f"{KD}/key_trivia*.json") for it in json.load(open(f))["items"]}
        tag = "trivia" if not prev else f"trivia{len(_g.glob(KD + '/key_trivia*.json')) + 1}"
        prev_match = sum(1 for f in _g.glob(f"{KD}/key_trivia*.json") for it in json.load(open(f))["items"] if it["alias_match"])
        uniq = {k: v for k, v in uniq.items() if k not in prev}
        nm = [k for k, v in uniq.items() if not v["match"]]
        mt = sorted([k for k, v in uniq.items() if v["match"]], key=lambda z: (int(z[0]), z[1]))
        rs = np.random.RandomState(31)
        mt_s = [mt[j] for j in rs.choice(len(mt), min(max(0, 100 - prev_match), len(mt)), replace=False)] if not prev else []
        items = sorted(nm, key=lambda z: (int(z[0]), z[1])) + mt_s
        perm = rs.permutation(len(items)); ids = rs.choice(np.arange(100000, 999999), len(items), replace=False)
        D = ROOT + "/blind_tc/" + tag; os.makedirs(D, exist_ok=True)
        rows, key = [], []
        for j, k in enumerate(perm):
            q, t = items[k]; x = data[int(q)]
            al = sorted({a for a in x["aliases"] if a == a.strip()}, key=len)[:12]
            rows.append({"id": int(ids[j]), "question": x["question"], "gold_answer": x["answer"], "gold_aliases": al, "answer": t})
            key.append({"id": int(ids[j]), "q": q, "answer": t, "runs": sorted(uniq[(q, t)]["runs"]), "alias_match": bool(uniq[(q, t)]["match"])})
        nch = int(np.ceil(len(rows) / 250)); bounds = np.linspace(0, len(rows), nch + 1).astype(int)
        for c in range(nch):
            json.dump(rows[bounds[c]:bounds[c + 1]], open(f"{D}/chunk_{c:02d}.json", "w"), indent=0)
        json.dump({"runs": P, "items": key}, open(f"{KD}/key_{tag}.json", "w"))
        print("trivia items", len(rows), "non-match", len(nm), "match sample", len(mt_s), nch, "chunks")

"""Stage G: graded-prompt generation + response-mean residuals at ALL layers.

usage: RUN=main python gen.py validate      wording check: 2 TRAIN questions per source trait x 25 cells x 1 sample (texts only)
       RUN=main python gen.py main          crossed design: all questions x 25 cells x S_PER_CELL samples
Per row: q, cell (trait, level), paraphrase p = sample mod 5, response ids/text, ntok, natural_end.
Activations (main only), float16 [N, 27, 2304], response tokens only, stop token excluded:
  acts_sys = the response read in its generation context (graded system text prepended to the user turn);
  acts_neu = the same response re-read in the NEUTRAL context (no system text)  <- PRIMARY (prereg section 3).
Sharded by question (resume-safe): <OUT>/gen_main/shard_XXX.{json,npz}; merged by merge().
"""
import os, sys, glob, numpy as np
from common import OUT, SMOKE, MAXNEW, S_PER_CELL, TRAITS, questions, cells, system_text, seed_of, jdump, jload, npsave, log
from lm import LM

STAGE = sys.argv[1]


def rows_for(qs, S):
    rows = []
    for qd in qs:
        for (t, l) in cells():
            for s in range(S):
                rows.append(dict(qid=qd["qid"], src=qd["src"], split=qd["split"], q=qd["q"], trait=t, level=l, p=s % 5, sample=s))
    return rows


def run_rows(lm, rows, tag, acts=True):
    ctx = [lm.chat_ids(system_text(r["trait"], r["level"], r["p"]) if r["trait"] != "none" else None, r["q"]) for r in rows]
    resps = lm.generate(ctx, seed_of(tag), MAXNEW)
    for r, x in zip(rows, resps):
        r["resp"] = x; r["text"] = lm.decode(x); r["ntok"] = len([t for t in x if t not in lm.stop])
        r["natural_end"] = bool(len(x) > 0 and x[-1] in lm.stop)
    if not acts:
        return rows, None, None
    L = list(range(lm.nl + 1))
    A_sys, _ = lm.tf(ctx, resps, layers=L)
    A_neu, _ = lm.tf([lm.chat_ids(None, r["q"]) for r in rows], resps, layers=L)
    return rows, A_sys, A_neu


def merge():
    d = f"{OUT}/gen_main"
    J = sorted(glob.glob(f"{d}/shard_*.json"))
    rows = [r for f in J for r in jload(f)]
    N = len(rows); first = np.load(J[0].replace(".json", ".npz"))
    shape = first["acts_neu"].shape[1:]
    for k in ("acts_sys", "acts_neu"):
        mm = np.lib.format.open_memmap(f"{OUT}/{k}.npy", mode="w+", dtype=np.float16, shape=(N,) + shape)
        i = 0
        for f in J:
            a = np.load(f.replace(".json", ".npz"))[k]; mm[i:i + len(a)] = a; i += len(a)
        mm.flush(); del mm
    for i, r in enumerate(rows): r["row"] = i
    jdump(rows, f"{OUT}/gen_main.json")
    log("merged", N, "rows", shape)


if __name__ == "__main__":
    if STAGE == "merge":
        merge(); sys.exit()
    lm = LM()
    if STAGE == "validate":
        qs = [q for q in questions() if q["split"] == "train"]
        nq = int(os.environ.get("NQ_VALIDATE", "1" if SMOKE else "2"))
        qs = [q for t in TRAITS for q in [x for x in qs if x["src"] == t][:nq]]
        rows = rows_for(qs, 1)
        for i, r in enumerate(rows): r["p"] = i % 5   # rotate paraphrases across the 1-sample cells
        rows, _, _ = run_rows(lm, rows, "validate", acts=False)
        for r in rows: r.pop("resp")
        jdump(rows, f"{OUT}/gen_validate.json"); log("validate rows", len(rows))
    elif STAGE == "main":
        os.makedirs(f"{OUT}/gen_main", exist_ok=True)
        qs = questions()   # Q_PER_TRAIT env (smoke: 4 = 2 TRAIN + 2 TEST per source trait)
        S = 2 if SMOKE else S_PER_CELL
        per = 2 if SMOKE else 8          # questions per shard
        for si in range(0, len(qs), per):
            f = f"{OUT}/gen_main/shard_{si // per:03d}.json"
            if os.path.exists(f): continue
            rows, A_sys, A_neu = run_rows(lm, rows_for(qs[si:si + per], S), f"main{si}")
            npsave(f.replace(".json", ".npz"), acts_sys=A_sys, acts_neu=A_neu)
            jdump(rows, f)
            log(f"shard {si // per}: {len(rows)} rows, mean ntok {np.mean([r['ntok'] for r in rows]):.0f}, nat {np.mean([r['natural_end'] for r in rows]):.2f}")
        merge()

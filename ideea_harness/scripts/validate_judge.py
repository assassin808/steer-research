"""[harness addition] Validate the proxy truth judges against TruthfulQA reference answers (leave-one-out).

For every question of a split, each reference correct answer (label 1) and each reference incorrect answer (label 0)
is judged as if it were a model answer, with that answer removed from the reference lists. Questions where the
removal would leave a list empty are skipped for that item. Reports accuracy, balanced accuracy, and per-judge AUC.
"I have no comment"-style references are skipped (they are refusals and judged truthful by rule).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng_lib as L  # noqa: E402
from common.io import load_json  # noqa: E402
from common.paths import dataset_dir  # noqa: E402
from common.proxy_judge import QwenJudge, is_refusal  # noqa: E402


def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(split="test", out=None, judges=("emb", "qwen"), max_q=0):
    df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
    ids = [int(i) for i in load_json(dataset_dir("truthfulqa") / "split.json")[split]]
    if max_q:
        ids = ids[:max_q]
    items = []
    for i in ids:
        good, bad = L.split_refs(df.loc[i, "Correct Answers"]), L.split_refs(df.loc[i, "Incorrect Answers"])
        for lab, lst in ((1, good), (0, bad)):
            for j, a in enumerate(lst):
                if is_refusal(a):
                    continue
                g2 = [x for k, x in enumerate(good) if not (lab == 1 and k == j)]
                b2 = [x for k, x in enumerate(bad) if not (lab == 0 and k == j)]
                if not g2 or not b2:
                    continue
                items.append((i, df.loc[i, "Question"], a, g2, b2, lab))
    y = np.array([it[-1] for it in items])
    res = {"split": split, "n_items": len(items), "n_questions": len(set(it[0] for it in items)), "share_true": float(y.mean())}
    preds = {}
    if "emb" in judges:
        ej = L.EmbJudge()
        sc = np.array([ej.truth_margin([it[2]], it[3], it[4])[0] for it in items])
        preds["emb"] = (sc > 0).astype(int)
        res["emb"] = {"acc": float((preds["emb"] == y).mean()),
                      "bal_acc": float(0.5 * ((preds["emb"][y == 1] == 1).mean() + (preds["emb"][y == 0] == 0).mean())),
                      "auc": auc(sc, y)}
    if "qwen" in judges:
        qj = QwenJudge()
        py = qj.p_yes([(it[1], it[2], it[3], it[4]) for it in items])
        preds["qwen"] = (py > 0.5).astype(int)
        res["qwen"] = {"acc": float((preds["qwen"] == y).mean()),
                       "bal_acc": float(0.5 * ((preds["qwen"][y == 1] == 1).mean() + (preds["qwen"][y == 0] == 0).mean())),
                       "auc": auc(py, y)}
    # threshold calibration (use only on a non-test split): best balanced accuracy
    def best_thr(score, y):
        grid = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
        ba = [0.5 * ((score[y == 1] > t).mean() + (score[y == 0] <= t).mean()) for t in grid]
        k = int(np.argmax(ba))
        return float(grid[k]), float(ba[k])
    if "emb" in judges:
        res["emb"]["best_thr"], res["emb"]["best_bal_acc"] = best_thr(sc, y)
    if "qwen" in judges:
        res["qwen"]["best_thr"], res["qwen"]["best_bal_acc"] = best_thr(py, y)
        np.save(os.path.splitext(out)[0] + "_qwen_scores.npy", py) if out else None
    if out and "emb" in judges:
        np.save(os.path.splitext(out)[0] + "_emb_scores.npy", sc)
        np.save(os.path.splitext(out)[0] + "_labels.npy", y)
    if len(preds) == 2:
        res["emb_qwen_agreement"] = float((preds["emb"] == preds["qwen"]).mean())
    print(json.dumps(res, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(split=sys.argv[1] if len(sys.argv) > 1 else "test", out=sys.argv[2] if len(sys.argv) > 2 else None)

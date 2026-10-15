"""[harness addition] Add Qwen-judge P(yes) to stored score-feature shards (runs_ng/*_shard*.npz -> *_qwen.npy)."""
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng_lib as L  # noqa
from common.paths import dataset_dir  # noqa
from common.proxy_judge import QwenJudge, clean  # noqa

df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
qj = QwenJudge()
for f in sorted(glob.glob(os.path.join(os.path.dirname(L.HERE), "runs_ng", "*_shard*.npz"))):
    out = f.replace(".npz", "_qwen.npy")
    if os.path.exists(out):
        continue
    z = np.load(f)
    items = [(df.loc[int(i), "Question"], clean(t) or "(empty)", L.split_refs(df.loc[int(i), "Correct Answers"]),
              L.split_refs(df.loc[int(i), "Incorrect Answers"])) for i, t in zip(z["qidx"], z["texts"])]
    np.save(out, qj.p_yes(items, bs=32))
    print(out, flush=True)

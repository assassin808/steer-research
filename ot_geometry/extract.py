"""Dump IDEEA's dev-split (train+val questions) per-head o_proj-input activations for Gemma-2-2B TruthfulQA.
Same rows IDEEA clusters on (task.probe_indices -> dev_df). Test questions are NOT loaded."""
import json, numpy as np, pandas as pd, sys
R = "/home/yang0531/scratch/ideea_gemma/results/truthfulqa"
df = pd.read_pickle(f"{R}/gemma2_2b/activations/head.pkl")
print(df.columns.tolist()[:5], len(df))
split = json.load(open(f"{R}/dataset/split.json"))
dev = set(split["development"])
d = df[df["question_idx"].isin(dev)].reset_index(drop=True)
L, H, D = 26, 8, 256
X = np.zeros((len(d), L, H, D), np.float32)
for l in range(L):
    X[:, l] = np.stack(d[f"model.layers.{l}.self_attn.o_proj"].to_numpy()).reshape(len(d), H, D)
lab = d["label"].to_numpy().astype(np.int8); q = d["question_idx"].to_numpy().astype(np.int32)
is_tr = np.isin(q, split["train"])
print("rows", len(d), "pos", (lab == 1).sum(), "neg", (lab == -1).sum(), "train rows", is_tr.sum(), "nq", len(set(q)))
# (full array not saved; per-head files below)
import os
os.makedirs("/home/yang0531/scratch/ot_geom/heads", exist_ok=True)
for l in range(L):
    for h in range(H):
        np.save(f"/home/yang0531/scratch/ot_geom/heads/L{l}H{h}.npy", X[:, l, h])
np.savez("/home/yang0531/scratch/ot_geom/meta.npz", label=lab, q=q, is_train=is_tr)

"""[harness addition] Same-supervision CAA control: direction = within-question Cov(s, mean layer-L residual over the
answer tokens), computed on the SAME judged base samples (train split, Qwen-judge s) used for our natural-gradient
direction. Isolates 'which data' from 'score-function/Fisher machinery'."""
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng_lib as L  # noqa
from ng_fit import load_split, chat_for, OUT  # noqa
from common.paths import dataset_dir  # noqa

layer = 14
df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
D = load_split("train", layer, "qwen")
tok, model = L.load_model(torch.bfloat16)
H = []
store = {}
hk = model.model.layers[layer].register_forward_hook(lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
with torch.no_grad():
    for k, (q, t) in enumerate(zip(D["qidx"], D["texts"])):
        pid = L.prompt_ids(tok, chat_for(df.loc[int(q), "Question"]))
        cid = torch.tensor(tok(str(t), add_special_tokens=False).input_ids or [tok.eos_token_id])
        ids = torch.cat([pid, cid])[None].to(L.DEV)
        model(ids)
        H.append(store["h"][0, pid.numel() - 1:].mean(0).cpu().numpy())
hk.remove()
H = np.stack(H).astype(np.float64)
s, q = D["s"].astype(np.float64), D["qidx"]
Hc, sc = np.empty_like(H), np.empty_like(s)
for i in np.unique(q):
    m = q == i
    Hc[m] = H[m] - H[m].mean(0); sc[m] = s[m] - s[m].mean()
v = Hc.T @ sc / len(s)
np.savez(f"{OUT}/scaa_L{layer}.npz", v=v)
Dq = np.load(f"{OUT}/directions_L{layer}_qwen.npz")
print("norm", np.linalg.norm(v), "cos(scaa, ng)", v @ Dq["ng"] / np.linalg.norm(v) / np.linalg.norm(Dq["ng"]),
      "cos(scaa, g)", v @ Dq["g"] / np.linalg.norm(v) / np.linalg.norm(Dq["g"]))

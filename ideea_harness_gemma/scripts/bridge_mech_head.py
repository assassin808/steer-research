import json, collections
import numpy as np
B = "/home/yang0531/scratch/ideea_gemma/runs_ng/bridge/"
fit = json.load(open(B + "fit_bridge.json")); HEADS = [tuple(x) for x in fit["heads"]]; HD = 256
iti = json.load(open("/home/yang0531/scratch/ideea_gemma/results/truthfulqa/gemma2_2b/directions/iti/directions.json"))
NS = np.array(iti["naive_std"]); STD = np.array([NS[l][h] for l, h in HEADS])
S = np.load(B + "samples_train.npz"); U = S["U"].astype(np.float64); q = S["qidx"]; toks = S["toks"]; offs = S["offs"]
seqs = [tuple(toks[offs[k]:offs[k + 1]]) for k in range(len(offs) - 1)]
c = collections.Counter(s[:5] for s in seqs)
print("top 5-token prefixes", c.most_common(3), "n", len(seqs))
top = c.most_common(1)[0][0]
ref = np.array([s[:5] == top for s in seqs])
print("refusal-prefix fraction of samples", ref.mean())
# within-question refusal-vs-answer score difference
dif = []
for i in np.unique(q):
    m = q == i
    if (m & ref).sum() and (m & ~ref).sum():
        dif.append(U[m & ref].mean(0) - U[m & ~ref].mean(0))
dif = np.array(dif); rdir = dif.mean(0)
print("questions with both", len(dif))
G = np.load(B + "GH.npy"); w, V = np.linalg.eigh(G); w, V = w[::-1], V[:, ::-1]
print("top eig", w[:5], "tr/d", w.sum() / len(w))
print("cos(refusal-score axis, top eigvecs 1..5)", [round(float(abs(V[:, k] @ rdir) / np.linalg.norm(rdir)), 3) for k in range(5)])
print("frac of refusal-axis energy in top-5 / top-20 eigvecs", float(((V[:, :5].T @ rdir) ** 2).sum() / (rdir @ rdir)), float(((V[:, :20].T @ rdir) ** 2).sum() / (rdir @ rdir)))
def applied(name):
    d = np.array(json.load(open(B + f"dirs/{name}.json"))["normalized_dir"])
    return np.concatenate([d[l, h] * s for (l, h), s in zip(HEADS, STD)])
print("%-7s  dlogodds(ref vs ans)/FisherNorm   frac Fisher-energy in top5   Fisher-norm per Euclid-norm  head-norm eff#" % "dir")
for n in ["B_inf", "B_10", "B_1", "C_inf", "C_10", "C_1", "G_inf", "G_10", "G_1", "H_inf", "H_10", "H_1", "E_inf"]:
    v = applied(n); fn = np.sqrt(v @ G @ v); pr = V[:, :5].T @ v
    hn = np.array([np.linalg.norm(v[j * HD:(j + 1) * HD]) for j in range(24)]); p = hn ** 2 / (hn ** 2).sum()
    print("%-7s  %+.3f                            %.3f                        %.4f                       %.1f" % (n, v @ rdir / fn, (w[:5] * pr ** 2).sum() / fn ** 2, fn / np.linalg.norm(v), 1 / (p ** 2).sum()))

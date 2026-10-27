"""Per-layer toolkit statistics + nulls. usage: python run_geom.py --model gemma --dom month [--k 10 --d 64 --reps 20 --label y]"""
import argparse, json, os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from multiprocessing import Pool
import geom as G

R = os.path.expanduser("~/scratch/weekday")


def one_layer(args):
    l, Hl, y, C, k, d, reps, seed, tm = args
    rng = np.random.default_rng(seed)
    X = G.pca(Hl.astype(np.float64), d)
    out = {"layer": l, "data": G.all_stats(X, y, C, k, rng, tmpl=tm)}
    nulls = {}
    for nm in ["flatG", "line", "line2", "blob", "shuffle", "resample"]:
        L = []
        for r in range(reps):
            if nm == "flatG": Xn, yn = G.null_flatG(X, y, rng)
            elif nm == "line": Xn, yn = G.null_line(X, y, C, rng)
            elif nm == "line2": Xn, yn = G.null_line2(X, y, C, rng)
            elif nm == "blob": Xn, yn = G.null_blob(X, y, C, rng)
            elif nm == "shuffle": Xn, yn = G.null_shuffle(X, y, rng)
            else: Xn, yn = G.null_resample(X, y, C, rng)
            L.append(G.all_stats(Xn, yn, C, k, rng, point=nm in ("flatG", "line", "line2", "blob"), tmpl=None if nm == "resample" else tm))
        nulls[nm] = {key: [x[key] for x in L] for key in L[0]}
    out["nulls"] = nulls
    print(f"L{l} id {out['data']['id']:.2f} (G {np.median(nulls['flatG']['id']):.1f}) dist {out['data']['distortion']:.3f} "
          f"(G {np.max(nulls['flatG']['distortion']):.3f} line {np.max(nulls['line']['distortion']):.3f}) "
          f"cycG {out['data']['cyc_G']:.2f} cycE {out['data']['cyc_E']:.2f} turn {out['data']['turn_naive']:.0f}/{out['data']['turn_corr']:.0f} "
          f"(line {np.median(nulls['line']['turn_corr']):.0f}) top1 {out['data']['top1']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma"); ap.add_argument("--dom", required=True)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--label", default="y", choices=["y", "pred", "correct", "z"])
    ap.add_argument("--layers", default="all"); ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    z = np.load(f"{R}/{a.model}_{a.dom}.npz")
    H, y, tm = z["H"], z["y"], z["t"]
    if a.label == "pred":
        y = z["pred"]
    elif a.label == "z":  # exploratory: the start day/month named in the prompt
        y = z["z"]
    elif a.label == "correct":
        m = z["pred"] == z["y"]; H, y, tm = H[m], y[m], tm[m]
    _, keep = np.unique(H[:, H.shape[1] // 2].round(4), axis=0, return_index=True)  # drop exact duplicates (z-token prefixes)
    keep = np.sort(keep)
    if len(keep) < len(y):
        print("dedup", len(y), "->", len(keep), flush=True); H, y, tm = H[keep], y[keep], tm[keep]
    C = 7 if a.dom == "weekday" else 12
    Ls = range(H.shape[1]) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    jobs = [(l, H[:, l], y, C, a.k, a.d, a.reps, 1000 + l, tm) for l in Ls]
    with Pool(a.procs) as p:
        res = p.map(one_layer, jobs)
    fn = f"{R}/geom_{a.model}_{a.dom}_k{a.k}_d{a.d}_{a.label}.json"
    json.dump(dict(args=vars(a), n=int(len(y)), counts=np.bincount(y, minlength=C).tolist(), layers=res), open(fn, "w"))
    print("wrote", fn)


if __name__ == "__main__":
    main()

"""Weekday/month calibration: extract last-token residual activations at all layers + optional steering.
Prereg: results/weekday_calib_prereg.md
usage: python extract.py --model gemma|qwen --out ~/scratch/weekday
"""
import argparse, copy, json, os, time
_PRE = {}
import numpy as np, torch
from scipy.interpolate import CubicSpline
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {"gemma": "google/gemma-2-2b-it", "qwen": "Qwen/Qwen2.5-1.5B-Instruct"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
T_DAY = ["What day is {k} {u} after {z}?",
         "What day of the week is {k} {u} after {z}?",
         "If today is {z}, what day will it be in {k} {u}?",
         "Starting from {z}, count forward {k} {u}. Which day is it?",
         "{k} {u} after {z} is which day of the week?",
         "What is the day {k} {u} after {z}?"]
T_MON = ["What month is {k} {u} after {z}?",
         "What month of the year is {k} {u} after {z}?",
         "If this month is {z}, what month will it be in {k} {u}?",
         "Starting from {z}, count forward {k} {u}. Which month is it?",
         "{k} {u} after {z} is which month of the year?",
         "What is the month {k} {u} after {z}?"]
PREFIX = "The answer is **"  # A1: Gemma emits " **" after "The answer is"; class tokens follow "**" without a space
STEER = {"weekday": [(0, 2), (0, 3)], "month": [(0, 2), (0, 3)]}
TGRID = np.linspace(0, 1, 21)


def build(dom):
    names, T, K, unit = (DAYS, T_DAY, range(1, 14), "day") if dom == "weekday" else (MONTHS, T_MON, range(1, 12), "month")
    rows = []
    for ti, t in enumerate(T):
        for zi, z in enumerate(names):
            for k in K:
                rows.append(dict(text=t.format(k=k, u=unit if k == 1 else unit + "s", z=z), t=ti, z=zi, k=k, y=(zi + k) % len(names)))
    return names, rows


def fmt(tok, q):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return s + PREFIX


def class_ids(tok, names):
    ids = []
    for n in names:
        a = tok.encode(n, add_special_tokens=False)
        ids.append(a[0])
        if len(a) != 1:
            print("WARN multi-token", n, a, flush=True)
    assert len(set(ids)) == len(ids), ids
    return ids


@torch.no_grad()
def forward(model, tok, texts, bs=16):
    """no padding: bucket by token length (positions stay exact)."""
    enc = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids[0] for t in texts]
    H = [None] * len(texts); P = [None] * len(texts)
    byL = {}
    for i, e in enumerate(enc):
        byL.setdefault(len(e), []).append(i)
    for L_, idx in byL.items():
        for j in range(0, len(idx), bs):
            b = idx[j:j + bs]
            out = model(torch.stack([enc[i] for i in b]), output_hidden_states=True)
            lp = torch.log_softmax(out.logits[:, -1].float(), -1)
            hs = torch.stack([h[:, -1].float() for h in out.hidden_states], 1)
            for r, i in enumerate(b):
                H[i] = hs[r].numpy(); P[i] = lp[r]
    return np.stack(H), torch.stack(P)


@torch.no_grad()
def steer_prompt(model, tok, text, layer_mod, D):
    """D: (B, d) deltas added at the last token output of layer_mod. Prefix KV is computed once."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    B = len(D)
    if text not in _PRE:
        _PRE.clear(); _PRE[text] = model(ids[:, :-1], use_cache=True).past_key_values
    pre = copy.deepcopy(_PRE[text]); pre.batch_repeat_interleave(B)
    def hk(mod, inp, outp):
        hs = outp[0] if isinstance(outp, tuple) else outp
        hs[:, -1] += D
        return outp
    h = layer_mod.register_forward_hook(hk)
    try:
        out = model(ids[:, -1:].repeat(B, 1), past_key_values=pre)
    finally:
        h.remove()
    return torch.log_softmax(out.logits[:, -1].float(), -1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="both", choices=["both", "extract", "steer", "ztok"]); ap.add_argument("--doms", default="weekday,month")
    a = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODELS[a.model]); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.set_num_threads(int(os.environ.get("NT", "12")))
    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], torch_dtype=torch.float32, attn_implementation="eager").eval()
    layers = model.model.layers
    os.makedirs(a.out, exist_ok=True)
    for dom in a.doms.split(","):
        names, rows = build(dom)
        cid = class_ids(tok, names)
        texts = [fmt(tok, r["text"]) for r in rows]
        fn = f"{a.out}/{a.model}_{dom}.npz"
        if a.mode == "ztok":  # A6: residual at the token naming the start day/month z (where Engels et al. report circles)
            zid = {i: set(tok.encode(" " + n, add_special_tokens=False)[:1] + tok.encode(n, add_special_tokens=False)[:1]) for i, n in enumerate(names)}
            pref = []
            for r, t in zip(rows, texts):
                ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids[0]
                pos = [p for p, v in enumerate(ids.tolist()) if v in zid[r["z"]]]
                assert len(pos) >= 1, (t, r)
                pref.append(ids[:pos[0] + 1])
            Hz = [None] * len(rows); byL = {}
            for i, e in enumerate(pref):
                byL.setdefault(len(e), []).append(i)
            with torch.no_grad():
                for L_, idx in byL.items():
                    for j in range(0, len(idx), 16):
                        b = idx[j:j + 16]
                        out = model(torch.stack([pref[i] for i in b]), output_hidden_states=True)
                        hs = torch.stack([h[:, -1].float() for h in out.hidden_states], 1)
                        for q, i in enumerate(b):
                            Hz[i] = hs[q].numpy()
            zz = dict(np.load(fn)); zz["H"] = np.stack(Hz).astype(np.float32)
            np.savez_compressed(f"{a.out}/{a.model}ztok_{dom}.npz", **zz)
            print(f"ztok {dom} done {time.time()-t0:.0f}s", flush=True)
            continue
        if a.mode == "steer":
            zz = np.load(fn); H, y, lpc = zz["H"], zz["y"], zz["lpc"]
        else:
            H, LP = forward(model, tok, texts)
            lpc = LP[:, cid].numpy()
            pred = lpc.argmax(1); y = np.array([r["y"] for r in rows])
            mass = np.exp(lpc).sum(1)
            print(f"{a.model} {dom} n={len(rows)} acc={np.mean(pred == y):.3f} class-mass med={np.median(mass):.3f} H{H.shape} {time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(fn, H=H.astype(np.float32), y=y, t=np.array([r["t"] for r in rows]),
                                z=np.array([r["z"] for r in rows]), k=np.array([r["k"] for r in rows]), pred=pred, lpc=lpc, mass=mass)
        if a.mode == "extract":
            continue

        # ---------------- steering
        tr = np.array([r["t"] for r in rows]) <= 4
        C = int(len(names))
        nl = H.shape[1] - 1
        Ls = list(range(4, min(nl - 2, 22) + 1, 2))
        if os.environ.get("STEER_LAYERS"):  # CPU budget (A7): subset of the preregistered layer grid
            Ls = [int(x) for x in os.environ["STEER_LAYERS"].split(",")]
        steer_rows = [i for i, r in enumerate(rows) if r["t"] == 5 and r["y"] == 0]
        stexts = [texts[i] for i in steer_rows]
        res = {"layers": Ls, "t": TGRID.tolist(), "n_prompts": len(stexts), "paths": {}}
        for l in Ls:
            cent = np.stack([H[tr & (y == c), l].mean(0) for c in range(C)])
            cs = CubicSpline(np.arange(C + 1), np.concatenate([cent, cent[:1]]), bc_type="periodic")
            for (i0, i1) in STEER[dom]:
                span = i1 - i0
                paths = {
                    "straight": np.stack([cent[i0] + s * (cent[i1] - cent[i0]) for s in TGRID]),
                    "polyline": np.stack([cent[i0 + int(min(np.floor(s * span), span - 1))] + (s * span - min(np.floor(s * span), span - 1)) *
                                          (cent[i0 + int(min(np.floor(s * span), span - 1)) + 1] - cent[i0 + int(min(np.floor(s * span), span - 1))])
                                          for s in TGRID]),
                    "spline": cs(i0 + TGRID * span),
                }
                for pn, G in paths.items():
                    if pn not in os.environ.get("STEER_PATHS", "straight,polyline,spline").split(","):
                        continue
                    D = torch.tensor(G - cent[i0], dtype=torch.float32)
                    Pc_all, oth, tic = [], [], []
                    for txt in stexts:
                        LPs = steer_prompt(model, tok, txt, layers[l - 1], D)  # hidden_states[l] = output of block l-1
                        Pc = np.exp(LPs[:, cid].numpy()); Pc_all.append(Pc); oth.append(1 - Pc.sum(1))
                        tic.append(np.isin(LPs.argmax(1).numpy(), cid))
                    Pc_all = np.stack(Pc_all)
                    if pn == "straight":  # t=0 => no steering; must reproduce the unsteered class probs
                        res.setdefault("sanity_t0_maxdiff", []).append(float(np.abs(Pc_all[:, 0] - np.exp(lpc[steer_rows])).max()))
                    res["paths"][f"L{l}_{i0}-{i1}_{pn}"] = dict(pc=Pc_all.mean(0).tolist(), pc_per_prompt=Pc_all.tolist(),
                                                                 other=np.mean(oth, 0).tolist(), top_is_class=np.mean(tic, 0).tolist(),
                                                                 dnorm=np.linalg.norm(G - cent[i0], axis=1).tolist())
            print(f"  steer L{l} done {time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(f"{a.out}/{a.model}_{dom}_steer.json", "w"))
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()

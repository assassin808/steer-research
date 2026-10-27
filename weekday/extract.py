"""Weekday/month calibration: extract last-token residual activations at all layers + optional steering.
Prereg: results/weekday_calib_prereg.md
usage: python extract.py --model gemma|qwen --out ~/scratch/weekday
"""
import argparse, json, os, time
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
PREFIX = "The answer is"
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
        a = tok.encode(" " + n, add_special_tokens=False)
        ids.append(a[0])
        if len(a) != 1:
            print("WARN multi-token", n, a, flush=True)
    assert len(set(ids)) == len(ids), ids
    return ids


@torch.no_grad()
def forward(model, tok, texts, bs=32, hook=None):
    H, P = [], []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model(**enc, output_hidden_states=hook is None)
        P.append(torch.log_softmax(out.logits[:, -1].float(), -1).cpu())
        if hook is None:
            H.append(torch.stack([h[:, -1].float() for h in out.hidden_states], 1).cpu())
    return (torch.cat(H).numpy() if H else None), torch.cat(P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODELS[a.model]); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
    layers = model.model.layers
    os.makedirs(a.out, exist_ok=True)
    for dom in ["weekday", "month"]:
        names, rows = build(dom)
        cid = class_ids(tok, names)
        texts = [fmt(tok, r["text"]) for r in rows]
        if dom == "weekday":
            print(repr(texts[0]), flush=True)
        H, LP = forward(model, tok, texts)
        lpc = LP[:, cid].numpy()
        pred = lpc.argmax(1); y = np.array([r["y"] for r in rows])
        mass = np.exp(lpc).sum(1)
        print(f"{a.model} {dom} n={len(rows)} acc={np.mean(pred == y):.3f} class-mass med={np.median(mass):.3f} H{H.shape} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(f"{a.out}/{a.model}_{dom}.npz", H=H.astype(np.float16), y=y, t=np.array([r["t"] for r in rows]),
                            z=np.array([r["z"] for r in rows]), k=np.array([r["k"] for r in rows]), pred=pred, lpc=lpc, mass=mass)

        # ---------------- steering
        tr = np.array([r["t"] for r in rows]) <= 4
        C = int(len(names))
        nl = H.shape[1] - 1
        Ls = list(range(4, min(nl - 2, 22) + 1, 2))
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
                    D = torch.tensor(G - cent[i0], dtype=torch.float32, device=model.device)
                    out = []
                    for ti in range(len(TGRID)):
                        d = D[ti]
                        def hk(mod, inp, outp, d=d):
                            hs = outp[0] if isinstance(outp, tuple) else outp
                            hs[:, -1] += d
                            return outp
                        h = layers[l - 1].register_forward_hook(hk)  # hidden_states[l] = output of block l-1
                        try:
                            _, LPs = forward(model, tok, stexts, hook=True)
                        finally:
                            h.remove()
                        Pc = np.exp(LPs[:, cid].numpy())
                        out.append(dict(pc=Pc.mean(0).tolist(), other=float((1 - Pc.sum(1)).mean()),
                                        top_is_class=float(np.isin(LPs.argmax(1).numpy(), cid).mean())))
                    res["paths"][f"L{l}_{i0}-{i1}_{pn}"] = out
            print(f"  steer L{l} done {time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(f"{a.out}/{a.model}_{dom}_steer.json", "w"))
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()

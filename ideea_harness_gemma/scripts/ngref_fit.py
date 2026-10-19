"""[harness addition] Judge-free natural-gradient direction 'ngref'.

Trait functional (no judge, no model samples for labels), teacher-forced with IDEEA's eval prompt:
  r(x, theta) = mean_{a+ in correct refs} log p(a+ | prompt(x), theta) - mean_{a- in incorrect refs} log p(a- | prompt(x), theta)
theta = residual added at layer L, positions P-1.. (ng_lib protocol).  g = mean over TRAIN questions of grad_theta r at 0.
Metric G and ridge lam are exactly those of the judged 'ng' fit (directions_L14_qwen.npz G, fit_L14_qwen.json lam_rel):
  ngref = (G + lam I)^-1 g,   eucref = g.
Variant 'E' appends the stop token (<|im_end|>) to every reference answer so the functional also scores ending the answer.
Only the TRAIN split's reference answers are used (the data IDEEA/ITI/CAA see); VAL is used for a diagnostic only.
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ng_lib as L  # noqa
from ng_fit import split_ids, chat_for, OUT  # noqa
from common.paths import dataset_dir  # noqa
LAYER = 14


def per_question_grads(model, tok, steer, df, ids, d, eos):
    G_plain, G_eos, meta = [], [], []
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    for n, i in enumerate(ids):
        q = df.loc[i, "Question"]
        pid = L.prompt_ids(tok, chat_for(q)).cuda()
        good, bad = L.split_refs(df.loc[i, "Correct Answers"]), L.split_refs(df.loc[i, "Incorrect Answers"])
        out = []
        for with_eos in (False, True):
            conts = []
            for a in good + bad:
                t = tok(a, add_special_tokens=False).input_ids + ([stop] if with_eos else [])
                conts.append(torch.tensor(t).cuda())
            U, lp = L.score_features(model, steer, pid, conts, d, bs=16)
            ng = len(good)
            out.append((U[:ng].mean(0) - U[ng:].mean(0), float(lp[:ng].mean() - lp[ng:].mean())))
        G_plain.append(out[0][0]); G_eos.append(out[1][0]); meta.append((int(i), len(good), len(bad), out[0][1], out[1][1]))
        if n % 50 == 0:
            print(n, len(ids), flush=True)
    return np.stack(G_plain), np.stack(G_eos), meta


if __name__ == "__main__":
    df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
    tok, model = L.load_model(torch.float32)
    steer = L.RowSteer(model, LAYER)
    d = model.config.hidden_size
    t0 = time.time()
    tr = per_question_grads(model, tok, steer, df, split_ids("train"), d, None)
    va = per_question_grads(model, tok, steer, df, split_ids("val"), d, None)
    print("grads", time.time() - t0)
    D = np.load(f"{OUT}/directions_L{LAYER}_qwen.npz")
    G = D["G"].astype(np.float64)
    fit = json.load(open(f"{OUT}/fit_L{LAYER}_qwen.json"))
    lam = fit["lam_rel"] * np.trace(G) / d
    res = {"lam_rel": fit["lam_rel"], "lam": lam, "n_train_q": len(tr[2]), "n_val_q": len(va[2]), "variants": {}}
    out = {"G": G}
    ng, euc = D["ng"], D["g"]
    for name, (gt, gv) in {"": (tr[0], va[0]), "E": (tr[1], va[1])}.items():
        g, gval = gt.mean(0).astype(np.float64), gv.mean(0).astype(np.float64)
        v = L.ridge_solve(G, g, lam)
        # split-half stability of g across train questions
        rng = np.random.RandomState(0); p = rng.permutation(len(gt)); h = len(p) // 2
        ga, gb = gt[p[:h]].mean(0), gt[p[h:]].mean(0)
        cs = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        r = {"norm_g": float(np.linalg.norm(g)), "cos_ngref_eucref": cs(v, g), "cos_ngref_ng": cs(v, ng), "cos_eucref_euc": cs(g, euc),
             "cos_ngref_ng_judged_euc": cs(v, euc), "split_half_cos_g": cs(ga, gb),
             # held-out first-order gain of r (val questions) per unit G-norm, for each direction
             "val_r_gain_per_Gnorm": {k: float(w @ gval / np.sqrt(w @ G @ w)) for k, w in
                                      {"ngref": v, "eucref": g, "ng_judged": ng, "euc_judged": euc}.items()},
             "train_r_gain_per_Gnorm": {k: float(w @ g / np.sqrt(w @ G @ w)) for k, w in
                                        {"ngref": v, "eucref": g, "ng_judged": ng, "euc_judged": euc}.items()}}
        res["variants"][name or "plain"] = r
        out["ngref" + name], out["eucref" + name] = v, g
    res["base_logratio_r_train_mean"] = float(np.mean([m[3] for m in tr[2]])); res["base_logratio_r_train_mean_E"] = float(np.mean([m[4] for m in tr[2]]))
    np.savez(f"{OUT}/directions_ref_L{LAYER}.npz", **out)
    json.dump(res, open(f"{OUT}/fit_ref_L{LAYER}.json", "w"), indent=1)
    os.makedirs(os.path.join(os.path.dirname(OUT), "results_ng"), exist_ok=True)
    json.dump(res, open(os.path.join(os.path.dirname(OUT), "results_ng", f"fit_ref_L{LAYER}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

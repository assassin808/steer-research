"""H1 v2: better-powered test of 'eta_hat / Var_p(s) predict per-question steering gain' (FLOW_FORMAL.md §2, §6).

Changes vs eta_pilot.py (see DEVLOG):
  - prompt/judge/intervention = Track 1 (IDEEA harness): IDEEA's chat prompt for Qwen3-1.7B, Qwen3-1.7B proxy judge (threshold calibrated on dev reference answers;
    embedding-judge scores stored as a secondary) s = truth AND info, theta added at layer 14 at positions >= last prompt token (IDEEA's CAA protocol);
  - 60 test-split questions with 0 < p_hat < 1 on 16 screening samples (screening samples are not reused);
  - per question: 128 fresh base samples for fitting (two halves of 64 for split-half reliability), 128 fresh base
    samples for the baseline score, and 128 eval samples per (direction, budget);
  - directions: per-question natural gradient restricted to a k=32 sketch (top-32 eigenvectors of the global
    within-question Fisher G from the train split, i.e. directions shared across questions), random, global CAA,
    global natural gradient (Track 1) -- all at EQUAL REALIZED KL: alpha calibrated with 32 samples (Rao-Blackwellised
    sequence KL), then rescaled (KL ~ alpha^2) before the 128-sample evaluation, whose KL is measured again;
  - KL = Rao-Blackwellised sequence KL: sum_t KL(p_theta(.|y<t) || p_0(.|y<t)) over y ~ p_theta (lower variance than
    the log-ratio estimator, which is also stored).
Usage: python h1v2.py run --nq 60 ; python h1v2.py analyze
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HARN = os.path.expanduser("~/steer_math/ideea_harness")
sys.path.insert(0, os.path.join(HARN, "scripts"))
sys.path.insert(0, HARN)
import ng_lib as L  # noqa: E402
from common.io import load_json  # noqa: E402
from common.paths import dataset_dir  # noqa: E402
from tasks.truthfulqa import _question_prompt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_h1v2")
LAYER = 14
K_SKETCH = 32
EPS = (0.05, 0.2)
DIRS = ("pq", "rand", "caa", "ngG")


def chat_for(q):
    return [{"role": "user", "content": _question_prompt(q)}, {"role": "assistant", "content": "A: "}]


def sketch_ridge(Z, s, lam_rel=1e-2):
    """NG direction restricted to the sketch: w = (Zc^T Zc/n + lam I)^-1 Zc^T sc / n."""
    Zc, sc = Z - Z.mean(0), s - s.mean()
    n, k = Zc.shape
    C = Zc.T @ Zc / n
    lam = lam_rel * np.trace(C) / k
    return np.linalg.solve(C + lam * np.eye(k), Zc.T @ sc / n)


def eta_cf(Z, s, seed=0, lam_rel=1e-2):
    """Cross-fitted eta_hat: two-fold held-out R^2 of s on sketch features, floored at 0."""
    if s.var() == 0:
        return float("nan")
    idx = np.random.RandomState(seed).permutation(len(s))
    hs = (idx[: len(s) // 2], idx[len(s) // 2:])
    r2 = []
    for a, b in (hs, hs[::-1]):
        if s[a].var() == 0 or s[b].var() == 0:
            continue
        w = sketch_ridge(Z[a], s[a], lam_rel)
        pred = s[a].mean() + (Z[b] - Z[a].mean(0)) @ w
        r2.append(1 - ((s[b] - pred) ** 2).sum() / ((s[b] - s[b].mean()) ** 2).sum())
    return max(0.0, float(np.mean(r2))) if r2 else float("nan")


def eta_cf_raw(Z, s, seed=0, lam_rel=1e-2):
    """Same without the floor (for reliability)."""
    if s.var() == 0:
        return float("nan")
    idx = np.random.RandomState(seed).permutation(len(s))
    hs = (idx[: len(s) // 2], idx[len(s) // 2:])
    r2 = []
    for a, b in (hs, hs[::-1]):
        if s[a].var() == 0 or s[b].var() == 0:
            continue
        w = sketch_ridge(Z[a], s[a], lam_rel)
        pred = s[a].mean() + (Z[b] - Z[a].mean(0)) @ w
        r2.append(1 - ((s[b] - pred) ** 2).sum() / ((s[b] - s[b].mean()) ** 2).sum())
    return float(np.mean(r2)) if r2 else float("nan")


def pick_questions(nq, seed=0):
    fs = sorted(glob.glob(f"{HARN}/runs_ng/test_L{LAYER}_shard*.npz"))
    from common.proxy_judge import qwen_threshold
    q = np.concatenate([np.load(f)["qidx"] for f in fs])
    info = np.concatenate([np.load(f)["info"] for f in fs])
    py = np.concatenate([np.load(f.replace(".npz", "_qwen.npy")) for f in fs])
    s = (info & (py > qwen_threshold())).astype(float)  # Qwen-judge screening scores
    ok = [int(i) for i in np.unique(q) if 0 < s[q == i].mean() < 1]
    rng = np.random.RandomState(seed)
    chosen = sorted(rng.choice(ok, size=min(nq, len(ok)), replace=False).tolist())
    return chosen, len(np.unique(q)), len(ok)


def judge_s(judge, df, i, texts):
    """Primary s = Qwen-judge truth AND info (threshold calibrated on dev refs); also returns the embedding-judge s."""
    ej, qj = judge
    args = (df.loc[i, "Question"], texts, L.split_refs(df.loc[i, "Correct Answers"]), L.split_refs(df.loc[i, "Incorrect Answers"]))
    t, f, _ = qj.judge(*args)
    te, fe, _ = ej.judge(*args)
    judge_s.last_emb = (te & fe).astype(np.float32)
    return (t & f).astype(np.float32)


def stage_run(a):
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(dataset_dir("truthfulqa") / "TruthfulQA.csv")
    qs, n_screen, n_ok = pick_questions(a.nq)
    json.dump({"questions": qs, "n_screened": n_screen, "n_with_var": n_ok}, open(f"{OUT}/questions.json", "w"))
    print(f"{n_ok}/{n_screen} screened test questions have 0<p<1; using {len(qs)}", flush=True)
    D = np.load(f"{HARN}/runs_ng/directions_L{LAYER}_qwen.npz")
    B = D["evecs"][:, :K_SKETCH]
    v_ngG = D["ng"]
    v_caa = np.array(load_json(f"{HARN}/results/truthfulqa/qwen3_1.7b/directions/caa/directions.json")["direction"][LAYER])
    tok, model = L.load_model(torch.float32)
    steer = L.RowSteer(model, LAYER)
    from common.proxy_judge import QwenJudge
    judge = (L.EmbJudge(), QwenJudge(model=model, tokenizer=tok))
    d = model.config.hidden_size
    for qi, i in enumerate(qs):
        path = f"{OUT}/q{i:04d}.json"
        if os.path.exists(path):
            continue
        t0 = time.time()
        torch.manual_seed(10_000 + i)
        pid = L.prompt_ids(tok, chat_for(df.loc[i, "Question"]))
        pidc = pid.cuda()
        # 1. fit samples + score features
        fit = L.sample(model, tok, pid, a.n_fit)
        s_fit = judge_s(judge, df, i, [tok.decode(c, skip_special_tokens=True) for c in fit])
        s_fit_emb = judge_s.last_emb
        U, lp = L.score_features(model, steer, pidc, [c.cuda() for c in fit], d)
        Z = U.astype(np.float64) @ B
        # 2. baseline eval samples
        base = L.sample(model, tok, pid, a.n_eval)
        base_texts = [tok.decode(c, skip_special_tokens=True) for c in base]
        s_base = judge_s(judge, df, i, base_texts)
        s_base_emb = judge_s.last_emb
        # 3. Euclidean / entropy features
        with torch.no_grad():
            hs = {}
            hk = model.model.layers[LAYER].register_forward_hook(lambda m, inp, out: hs.__setitem__("h", (out[0] if isinstance(out, tuple) else out)[0, -1].float().cpu().numpy()))
            lg = model(pidc[None]).logits[0, -1].float()
            hk.remove()
            p0 = torch.softmax(lg, -1)
            ent0 = float(-(p0 * torch.log(p0 + 1e-30)).sum())
        h = hs["h"]
        # per-token entropy along base samples (NBF-style proxy): mean over samples of mean token entropy
        with torch.no_grad():
            lgs, _, tm = L.cont_logits(model, steer, pidc, [c.cuda() for c in base[:32]], None)
            lps = torch.log_softmax(lgs, -1)
            ent_tok = float(((-(lps.exp() * lps).sum(-1) * tm).sum(1) / tm.sum(1)).mean())
            del lgs, lps
        Uc = U - U.mean(0)
        Gx = (Uc.T.astype(np.float64) @ Uc.astype(np.float64)) / len(U)
        # directions
        wz = sketch_ridge(Z, s_fit)
        v_pq = B @ wz
        v_rand = np.random.RandomState(i).randn(d)
        dirs = {"pq": v_pq, "rand": v_rand, "caa": v_caa, "ngG": v_ngG}
        rec = {"q": i, "question": df.loc[i, "Question"], "p_fit": float(s_fit.mean()), "var_fit": float(s_fit.var()),
               "p_base": float(s_base.mean()), "p_fit_emb": float(s_fit_emb.mean()), "p_base_emb": float(s_base_emb.mean()),
               "eta_k_all_embjudge": eta_cf(Z, s_fit_emb), "var_base": float(s_base.var()), "n_fit": len(s_fit), "n_eval": len(s_base),
               "eta_k_all": eta_cf(Z, s_fit), "eta_k_A": eta_cf(Z[:64], s_fit[:64]), "eta_k_B": eta_cf(Z[64:], s_fit[64:]),
               "eta_k_all_raw": eta_cf_raw(Z, s_fit), "eta_k_A_raw": eta_cf_raw(Z[:64], s_fit[:64]), "eta_k_B_raw": eta_cf_raw(Z[64:], s_fit[64:]),
               "eta_k_all_sd": [eta_cf(Z, s_fit, seed=r) for r in range(5)],
               "ent_first": ent0, "ent_tok": ent_tok, "h_norm": float(np.linalg.norm(h)),
               "cos_h_caa": float(h @ v_caa / np.linalg.norm(h) / np.linalg.norm(v_caa)),
               "cos_h_ngG": float(h @ v_ngG / np.linalg.norm(h) / np.linalg.norm(v_ngG)),
               "cos_pq_caa": float(v_pq @ v_caa / np.linalg.norm(v_pq) / np.linalg.norm(v_caa)),
               "cos_pq_ngG": float(v_pq @ v_ngG / np.linalg.norm(v_pq) / np.linalg.norm(v_ngG)),
               "mean_ntok_base": float(np.mean([c.numel() for c in base])), "steer": {}}
        for other in (8, 16, 64, 128, 256):
            Zo = U.astype(np.float64) @ D["evecs"][:, :other]
            rec[f"eta_k{other}_all"] = eta_cf(Zo, s_fit)
            rec[f"eta_k{other}_A"] = eta_cf(Zo[:64], s_fit[:64]); rec[f"eta_k{other}_B"] = eta_cf(Zo[64:], s_fit[64:])
        for name, v in dirs.items():
            v = v / np.linalg.norm(v)
            gv = float(v @ (Uc.T @ (s_fit - s_fit.mean())) / len(U))  # first-order gain per unit alpha (sample)
            vGv = float(v @ Gx @ v)
            rec[f"pred_eff_{name}"] = gv / np.sqrt(max(vGv, 1e-30))  # predicted dT/ds (in-sample, optimistic for pq)
            for eps in EPS:
                a0 = float(np.sqrt(2 * eps / max(vGv, 1e-30)))
                th0 = torch.tensor(a0 * v, dtype=torch.float32, device="cuda")
                cal = L.sample(model, tok, pid, a.n_cal, steer=steer, theta=th0)
                kl0, _ = L.seq_kl(model, steer, pidc, [c.cuda() for c in cal], th0)
                fac = float(np.clip(np.sqrt(eps / max(kl0.mean(), 1e-6)), 0.25, 4.0))
                a1 = a0 * fac
                th1 = torch.tensor(a1 * v, dtype=torch.float32, device="cuda")
                ev = L.sample(model, tok, pid, a.n_eval, steer=steer, theta=th1)
                s_ev = judge_s(judge, df, i, [tok.decode(c, skip_special_tokens=True) for c in ev])
                s_ev_emb = judge_s.last_emb
                rb, mc = L.seq_kl(model, steer, pidc, [c.cuda() for c in ev], th1)
                rec["steer"][f"{name}@{eps}"] = {
                    "alpha0": a0, "kl_cal": float(kl0.mean()), "alpha": a1, "score": float(s_ev.mean()),
                    "gain": float(s_ev.mean() - s_base.mean()),
                    "gain_se": float(np.sqrt(s_ev.var() / len(s_ev) + s_base.var() / len(s_base))),
                    "gain_emb": float(s_ev_emb.mean() - s_base_emb.mean()),
                    "gain_A": float(s_ev[:64].mean() - s_base[:64].mean()), "gain_B": float(s_ev[64:].mean() - s_base[64:].mean()),
                    "kl_rb": float(rb.mean()), "kl_rb_se": float(rb.std() / np.sqrt(len(rb))), "kl_mc": float(mc.mean()),
                    "kl_mc_se": float(mc.std() / np.sqrt(len(mc))), "pred_kl_alpha1": float(0.5 * a1 ** 2 * vGv),
                    "mean_ntok": float(np.mean([c.numel() for c in ev]))}
        np.savez(f"{OUT}/q{i:04d}.npz", U=U, s_fit=s_fit, s_base=s_base, s_fit_emb=s_fit_emb, s_base_emb=s_base_emb, lp=lp)
        json.dump(rec, open(path, "w"), indent=1)
        print(f"[{qi + 1}/{len(qs)}] q{i} p_fit={rec['p_fit']:.2f} p_base={rec['p_base']:.2f} eta_k={rec['eta_k_all']:.2f} "
              + " ".join(f"{k}:{v['gain']:+.3f}/{v['kl_rb']:.3f}" for k, v in rec["steer"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--nq", type=int, default=60)
    ap.add_argument("--n_fit", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--n_cal", type=int, default=32)
    a = ap.parse_args()
    {"run": stage_run}[a.stage](a)

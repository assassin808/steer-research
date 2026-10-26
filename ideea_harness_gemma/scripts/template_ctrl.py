"""[harness addition, 2026-10-26] Template controls + TriviaQA check (prereg ~/steer_math/results/template_ctrl_prereg.md).

stages
  feats   (GPU, fp32) template data from TRAIN-question dev outputs: debunk answers of C_b1-b3 (dev strength 2.2954, regex TN) and pure
          abstentions of dev base / IDEEA s0 a4; per answer: eval-context head inputs averaged over answer positions (cH, as bridge feats)
          and head score features U (as bridge refs) -> runs_ng/template_ctrl/feats.npz
  fit     (CPU) T_md (mean-diff of cH) and T_g (gradient = mean-diff of U), within-question estimator, per-head unit, ITI file format;
          cosines vs C_orig / C_b* / IDEEA-ITI mass-mean -> runs_ng/template_ctrl/fit.json
  eval    (GPU) generation for 'tc:NAME:S' specs (and any gm_eval / bridge spec) on --split development|test|triviaqa
  forced  (GPU) base model with a forced assistant opener (--opener), test or triviaqa; KL_forced reported
  cfor    strengths for target dev KLs (bridge.c_for rule) from the dev sweep
"""
import argparse, json, os, re, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(HERE)
import bridge as B  # noqa: E402

TC = os.path.join(ROOT, "runs_ng", "template_ctrl"); TDIRS = os.path.join(TC, "dirs"); os.makedirs(TDIRS, exist_ok=True)
TQA = os.path.expanduser("~/scratch/tqa_ctrl/data/triviaqa400.json")
TN = re.compile(r"^\s*(there (is|are|'s) no|there isn't|there aren't|there is not|this is a (bit of a )?(trick|hypothetical|subjective)|it'?s (impossible|difficult) to)", re.I)
full = lambda t: (t.strip()[2:].strip() if t.strip().startswith("A:") else t.strip())
T_GRID = [0.5, 1, 2, 4, 8]
LEVELS = {"lo": 0.30, "mid": 0.518, "hi": 0.80}


def template_data():
    from gm_fix_select import is_ref
    tr = set(B.split_of("train"))
    D, A = {}, {}
    for s in (1, 2, 3):
        for r in json.load(open(f"{ROOT}/results_gm/development_confirm/br_C_inf_b{s}_2.2954.json"))["records"]:
            t = full(r["response"])
            if r["key"] in tr and TN.match(t):
                D.setdefault(int(r["key"]), set()).add(t)
    for f in ("development_grid/base.json", "development_confirm/br_A_kms0_4.json"):
        for r in json.load(open(f"{ROOT}/results_gm/{f}"))["records"]:
            t = full(r["response"])
            if r["key"] in tr and is_ref(t):
                A.setdefault(int(r["key"]), set()).add(t)
    Q = sorted(set(D) & set(A))
    return Q, {q: sorted(D[q]) for q in Q}, {q: sorted(A[q]) for q in Q}


def stage_feats(a):
    import torch
    import ng_lib as L
    import pandas as pd
    from gm_fit import ref_conts, chat_for
    df = pd.read_csv(B.CSV)
    Q, D, A = template_data()
    print("template questions", len(Q), "debunk", sum(map(len, D.values())), "abst", sum(map(len, A.values())), flush=True)
    tok, model = L.load_model(torch.float32)
    hs = B.HeadSteer(model)
    d = len(B.HEADS) * B.HD
    rec = {k: [] for k in ("U", "cH", "qidx", "debunk", "ntok")}
    texts = []
    for n, i in enumerate(Q):
        q = df.loc[i, "Question"]; pid = L.prompt_ids(tok, chat_for(q))
        ans = D[i] + A[i]
        conts, mism = ref_conts(tok, pid, q, ans)
        conts = [c.cuda() for c in conts]
        U, _ = L.score_features(model, hs, pid.cuda(), conts, d, bs=4)
        kv = L._prefix_cache(model, pid.cuda())
        cH = []
        for b in range(0, len(conts), 8):
            cs = conts[b:b + 8]
            hs.capture = True
            with torch.no_grad():
                _, _, tm = L.cont_logits(model, hs, pid.cuda(), cs, None, kv)
            hs.capture = False
            Ah = hs.gather(); w = tm / tm.sum(1, keepdim=True)
            cH.append(torch.einsum("bt,btd->bd", w, Ah).cpu().numpy())
        rec["U"].append(U); rec["cH"].append(np.concatenate(cH)); rec["qidx"].append(np.full(len(ans), i))
        rec["debunk"].append(np.array([1] * len(D[i]) + [0] * len(A[i]), bool)); rec["ntok"].append(np.array([c.numel() for c in conts]))
        texts += ans
        if n % 40 == 0:
            print("feats", n, flush=True)
    np.savez(f"{TC}/feats.npz", text=np.array(texts), **{k: np.concatenate(v).astype(np.float32) if k in ("U", "cH") else np.concatenate(v) for k, v in rec.items()})
    print("saved", flush=True)


def within_q(F, qidx, deb):
    out = []
    for q in np.unique(qidx):
        m = qidx == q
        out.append(F[m & deb].astype(np.float64).mean(0) - F[m & ~deb].astype(np.float64).mean(0))
    return np.mean(out, 0)


def save_dir(name, g):
    import bridge as B_
    old = B_.DIRS
    B_.DIRS = TDIRS
    try:
        B_.save_iti_format(name, [B_.unit(x) for x in B_.blocks(g)])
    finally:
        B_.DIRS = old


def stage_fit(a):
    Z = dict(np.load(f"{TC}/feats.npz"))
    deb = Z["debunk"].astype(bool)
    res = {"n_q": int(len(np.unique(Z["qidx"]))), "n_debunk": int(deb.sum()), "n_abst": int((~deb).sum()), "cos": {}}
    G = {"T_md": within_q(Z["cH"], Z["qidx"], deb), "T_g": within_q(Z["U"], Z["qidx"], deb)}
    for k, g in G.items():
        save_dir(k, g)
    np.savez(f"{TC}/raw_dirs.npz", **G)
    # reference vectors: C (bridge refs gradient, identity + bootstraps), IDEEA/ITI-style mass-mean of IDEEA's head data at the same heads
    import confirm_grad as CG
    R = dict(np.load(f"{B.OUT}/refs.npz"))
    refs = {"C_orig": CG.grad(R["U"], CG.draws(R, 0))}
    for b in CG.SEEDS:
        refs[f"C_b{b}"] = CG.grad(R["U"], CG.draws(R, b))
    iti = json.load(open(B.ITI_PATH))
    nd = np.array(iti["normalized_dir"]) if "normalized_dir" in iti else None
    if nd is not None:
        refs["ITI_massmean_unit_perhead"] = np.concatenate([nd[l, h] for l, h in B.HEADS])
    for k, g in G.items():
        for r, v in refs.items():
            res["cos"][f"{k}|{r}"] = {"global": B.cos(g, v), "perhead_eff": B.cos(B.ph_effective(g), B.ph_effective(v)),
                                       "perhead_mean": float(np.mean([B.cos(x, y) for x, y in zip(B.blocks(g), B.blocks(v))]))}
    res["cos"]["T_md|T_g"] = {"global": B.cos(G["T_md"], G["T_g"]), "perhead_eff": B.cos(B.ph_effective(G["T_md"]), B.ph_effective(G["T_g"]))}
    # split-half reliability of each template direction (questions split in two halves)
    qs = np.unique(Z["qidx"]); rs = np.random.RandomState(0); sh = []
    for _ in range(20):
        p = rs.permutation(qs); h1 = np.isin(Z["qidx"], p[:len(p) // 2])
        sh.append([B.cos(B.ph_effective(within_q(Z[f][h1], Z["qidx"][h1], deb[h1])),
                         B.ph_effective(within_q(Z[f][~h1], Z["qidx"][~h1], deb[~h1]))) for f in ("cH", "U")])
    res["split_half_perhead_eff"] = {"T_md": float(np.mean([x[0] for x in sh])), "T_g": float(np.mean([x[1] for x in sh]))}
    json.dump(res, open(f"{TC}/fit.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def tqa_requests():
    from common.modeling import GenRequest
    from gm_fit import chat_for
    data = json.load(open(TQA))
    return [GenRequest(key=j, chat=chat_for(x["question"]), meta={"question": x["question"], "tqa_id": x["id"]}) for j, x in enumerate(data)]


def make_ctx(split):
    E = B.patch_eval()
    orig = E.build

    def build(ctx, spec):
        if spec.startswith("tc:"):
            p = spec.split(":"); s = float(p[2])
            return ("ideea_hooks", B.FileITI(f"{TDIRS}/{p[1]}.json", s)), {"method": "tc", "name": p[1], "strength": s}
        return orig(ctx, spec)
    E.build = build
    ctx = E.Ctx("test" if split == "triviaqa" else split)
    if split == "triviaqa":
        ctx.requests = tqa_requests()
    return E, ctx


def stage_eval(a):
    import torch
    import ng_lib as L
    E, ctx = make_ctx(a.split)
    tok, model = L.load_model(torch.bfloat16)
    od = os.path.join(ROOT, "results_gm", a.split + "_" + a.tag); os.makedirs(od, exist_ok=True)
    specs = list(a.specs) + [f"tc:{nm}:{c:g}" for nm in a.calib for c in T_GRID]
    for s in specs:
        E.run_spec(ctx, model, tok, {}, s, a.bs, od)


def stage_forced(a):
    """base model, assistant message 'A: ' + opener (continue_final_message), greedy 50 new tokens.
    KL_forced = sum over opener tokens of -log p_base(token | prefix) / (n opener tokens + n generated tokens up to stop)."""
    import torch
    import ng_lib as L
    from common.config import deterministic
    E, ctx = make_ctx(a.split)
    tok, model = L.load_model(torch.bfloat16)
    od = os.path.join(ROOT, "results_gm", a.split + "_" + a.tag); os.makedirs(od, exist_ok=True)
    name = "forced_" + re.sub(r"[^A-Za-z]+", "_", a.opener).strip("_")
    path = f"{od}/{name}.json"
    if os.path.exists(path):
        print("exists", path); return
    deterministic(0)
    stops = L.stop_ids(tok); records = []; t0 = time.time()
    reqs = ctx.requests
    for b0 in range(0, len(reqs), a.bs):
        batch = reqs[b0:b0 + a.bs]
        chats = [[r.chat[0], {"role": "assistant", "content": "A: " + a.opener}] for r in batch]
        tails = []
        for r, c in zip(batch, chats):
            p0 = L.prompt_ids(tok, r.chat).tolist(); p1 = L.prompt_ids(tok, c).tolist()
            k = 0
            while k < min(len(p0), len(p1)) and p0[k] == p1[k]:
                k += 1
            tails.append(len(p1) - k)
        enc = tok.apply_chat_template(chats, padding="longest", continue_final_message=True, return_dict=True, return_tensors="pt").to("cuda")
        P = enc["input_ids"].shape[1]
        with torch.no_grad():
            lg = torch.log_softmax(model(**enc).logits.float(), -1)
            out = model.generate(**enc, max_new_tokens=ctx.task.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        ids = enc["input_ids"]
        for i, (r, nt) in enumerate(zip(batch, tails)):
            nll = float(-sum(lg[i, t - 1, ids[i, t]].item() for t in range(P - nt, P)))
            gen = out[i, P:].tolist()
            n = next((j + 1 for j, t in enumerate(gen) if t in stops), len(gen))
            text = tok.decode(out[i, P - nt:P + n].tolist(), skip_special_tokens=True)
            records.append({"key": r.key, "meta": r.meta, "response": ctx.task.parse(text), "kl_tok": nll / (nt + n), "kl_seq": nll,
                            "n_opener_tok": nt, "ntok": int(n)})
    kl = np.array([x["kl_tok"] for x in records])
    summ = {"spec": name, "method": "forced", "opener": a.opener, "n": len(records), "kl_tok_mean": float(kl.mean()),
            "kl_seq_mean": float(np.mean([x["kl_seq"] for x in records])), "mean_ntok": float(np.mean([x["ntok"] for x in records])),
            "gen_seconds": time.time() - t0}
    json.dump({"summary": summ, "records": records}, open(path, "w"))
    print(json.dumps(summ), flush=True)


def c_for(name, targets, tag="tc"):
    od = os.path.join(ROOT, "results_gm", "development_" + tag)
    pts = [(c, json.load(open(f"{od}/tc_{name}_{c:g}.json"))["summary"]["kl_tok_mean"]) for c in T_GRID]
    out = []
    for T in targets:
        cc = None
        for (c0, k0), (c1, k1) in zip(pts[:-1], pts[1:]):
            if k0 <= T <= k1 and k1 > k0:
                cc = float(np.exp(np.log(c0) + (np.log(T) - np.log(max(k0, 1e-5))) / (np.log(k1) - np.log(max(k0, 1e-5))) * (np.log(c1) - np.log(c0)))); break
        if cc is None:
            (c0, k0), (c1, k1) = (pts[0], pts[1]) if T < pts[0][1] else (pts[-2], pts[-1])
            sl = (np.log(max(k1, 1e-5)) - np.log(max(k0, 1e-5))) / (np.log(c1) - np.log(c0))
            cc = float(np.clip(c1 * np.exp((np.log(T) - np.log(max(k1, 1e-5))) / max(sl, 0.5)), 0.01, 64.0))
        out.append(round(cc, 4))
    return out, pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "fit", "eval", "forced", "cfor"])
    ap.add_argument("--specs", nargs="*", default=[]); ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--split", default="test"); ap.add_argument("--tag", default="tc"); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--opener", default="There is no"); ap.add_argument("--names", nargs="*", default=["T_md", "T_g"])
    a = ap.parse_args()
    if a.stage == "cfor":
        res = {}
        for nm in a.names:
            cs_, pts = c_for(nm, list(LEVELS.values()))
            res[nm] = {"scan": pts, "strengths": dict(zip(LEVELS, cs_))}
        json.dump(res, open(f"{TC}/strengths.json", "w"), indent=1)
        print(json.dumps(res))
    else:
        {"feats": stage_feats, "fit": stage_fit, "eval": stage_eval, "forced": stage_forced}[a.stage](a)

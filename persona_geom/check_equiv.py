"""Numerical checks of the steering hooks (CPU OK, ~minutes):
1. IDEEA rule: logits recorded during HF generate (hook at the last position of each forward call) == logits of one teacher-forced
   pass with the hook applied at every position >= last prompt token (lm.py docstring claim).
2. 'add' rule: same equivalence (trivial, but checks left-padding + broadcasting).
Uses random centroids at layer 8 with a large alpha so the edit matters. Prints max |logit difference|.
"""
import numpy as np, torch
from lm import LM, DEV

lm = LM(); L = 8; H = lm.H
rng = np.random.default_rng(0)
qs = ["How should I handle a coworker who takes credit for my work?", "What is a good way to start saving money?"]
ctxs = [lm.chat_ids(None, q) for q in qs]
pos_c, neg_c = rng.standard_normal((4, H)) * 3, rng.standard_normal((4, H)) * 3
for mode in ("ideea", "add"):
    S = max(len(c) for c in ctxs)
    ids = torch.full((2, S), lm.pad, dtype=torch.long); mask = torch.zeros_like(ids)
    for j, c in enumerate(ctxs): ids[j, S - len(c):] = torch.tensor(c); mask[j, S - len(c):] = 1
    D = [rng.standard_normal(H).astype(np.float32) * 4 for _ in ctxs]
    if mode == "ideea": lm.set_ideea(L, pos_c, neg_c, [2.0, 2.0])
    else: lm.set_add(L, D)
    torch.manual_seed(0)
    g = lm.model.generate(input_ids=ids.to(DEV), attention_mask=mask.to(DEV), do_sample=True, max_new_tokens=6, top_k=0,
                          eos_token_id=lm.stop, pad_token_id=lm.pad, output_logits=True, return_dict_in_generate=True)
    lm.clear()
    gl = torch.stack(g.logits, 1).float()                    # [2, T, V]
    resps = [g.sequences[j, S:].tolist() for j in range(2)]
    fr = torch.tensor([len(c) - 1 for c in ctxs], device=DEV)
    def st(ch):
        if mode == "ideea": lm.set_ideea(L, pos_c, neg_c, [2.0] * len(ch), from_pos=fr[ch])
        else: lm.set_add(L, [D[i] for i in ch])
    _, lps = lm.tf(ctxs, resps, want_lp=True, steer=st)
    _, lp0 = lm.tf(ctxs, resps, want_lp=True)
    for j in range(2):
        T = len(resps[j]); ref = torch.log_softmax(gl[j, :T], -1)
        print(mode, "row", j, "max|dlogp| gen vs TF-steered", float((ref - lps[j][:T]).abs().max()),
              " (vs unsteered:", float((ref - lp0[j][:T]).abs().max()), ")", flush=True)

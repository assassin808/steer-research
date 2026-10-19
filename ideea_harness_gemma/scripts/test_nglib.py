import os, sys, torch, numpy as np
sys.path.insert(0, os.path.expanduser("~/steer_math/ideea_harness/scripts"))
import ng_lib as L
tok, model = L.load_model(torch.float32)
st = L.RowSteer(model, 14)
chat=[{"role":"user","content":"Q: What is the capital of France?"},{"role":"assistant","content":"A: "}]
pid = L.prompt_ids(tok, chat)
torch.manual_seed(0)
conts = L.sample(model, tok, pid, 3, max_new=12)
print([tok.decode(c) for c in conts])
d = model.config.hidden_size
U, lp = L.score_features(model, st, pid, conts, d, bs=2)
# full-sequence reference logp with theta = a*v at positions >= P-1
v = torch.randn(d); v = v / v.norm()
pid = pid.to(L.DEV); conts = [c.to(L.DEV) for c in conts]; v = v.to(L.DEV)
def full_lp(c, a):
    ids = torch.cat([pid, c])[None]
    P = pid.numel()
    st.mode, st.start, st.theta = "from", P - 1, a * v
    lg = model(ids).logits[0, P-1:-1].float(); st.theta=None
    return torch.log_softmax(lg, -1).gather(1, c.to(L.DEV)[:, None]).sum().item()
with torch.no_grad():
    for k, c in enumerate(conts):
        e = 1e-2
        fd = (full_lp(c, e) - full_lp(c, -e)) / (2 * e)
        print("lp cached", lp[k], "full", full_lp(c, 0.0), "| grad.v", float(U[k] @ v.cpu().numpy()), "fd", fd)
    th = 0.5 * v
    rb, mc = L.seq_kl(model, st, pid, conts, th)
    print("kl", rb, mc)
# generation with last-mode equals teacher-forced 'from' mode?
st.mode, st.theta = "last", 3 * v
out = model.generate(pid[None], attention_mask=torch.ones_like(pid[None]), max_new_tokens=8, do_sample=False, output_logits=True, return_dict_in_generate=True)
st.theta=None
g = out.sequences[0, pid.numel():]
lg, tgt, tm = L.cont_logits(model, st, pid, [g], 3 * v)
print("gen vs tf max abs diff", (torch.stack(out.logits,1)[0].float() - lg[0, :len(g)]).abs().max().item())

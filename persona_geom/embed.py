"""Surface-form control: sentence-encoder embeddings of the RESPONSE TEXT (all-mpnet-base-v2, mean pooling + L2 norm,
i.e. the sentence-transformers recipe for this model, re-implemented with plain transformers). CPU is fine.
usage: RUN=main python embed.py gen_main [judged_steer ...]  -> <OUT>/emb_<tag>.npy  [N, 768] float32
"""
import os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from common import EMB_MODEL, OUT, jload, log

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(EMB_MODEL); m = AutoModel.from_pretrained(EMB_MODEL).to(dev).eval()
    for tag in sys.argv[1:]:
        dst = f"{OUT}/emb_{tag}.npy"
        if os.path.exists(dst): continue
        texts = [r["text"] for r in jload(f"{OUT}/{tag}.json")]
        E = []
        with torch.no_grad():
            for b in range(0, len(texts), 64):
                enc = tok(texts[b:b + 64], padding=True, truncation=True, max_length=384, return_tensors="pt").to(dev)
                h = m(**enc).last_hidden_state
                mk = enc.attention_mask[..., None].float()
                e = (h * mk).sum(1) / mk.sum(1).clamp(min=1e-9)
                E.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        np.save(dst, np.concatenate(E).astype(np.float32)); log("emb", tag, len(texts))

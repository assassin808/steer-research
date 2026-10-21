"""[harness addition] Test specs at matched dev KL 0.518 (prereg §5) from bridge_selection.json. Prints specs (stdout)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(ROOT + "/bridge_selection.json"))["families"]
specs = []
def add(cfg, c):
    if cfg.startswith("K_"):
        k = {"K_0.1": "nfsum0.1", "K_1": "nfsum1"}.get(cfg)
        if k:
            specs.append(f"fx:{k}:13:{c:g}")
    else:
        specs.append(f"br:{cfg}:{c:g}")
for f, v in S.items():
    m = v["matched"]
    if f == "F":
        dev = v["dev_sweeps"]
        best = lambda cs: max(cs, key=lambda c: max([x[2] for x in dev[c] if x[1] <= 1.2 and x[3] <= json.load(open(ROOT + "/bridge_selection.json"))["base_dev"]["refusal"]] or [-1]))
        loc_plain = best([c for c in m if c.endswith("_inf") and not c.startswith("F_inf")])
        want = {v["selected"]["config"] if v["selected"] else None, "F_inf_inf", "F_inf_1", loc_plain, (v["best_nonplain"] or {}).get("config")}
        for c in sorted(x for x in want if x):
            add(c, m[c])
    else:
        for c, s in m.items():
            add(c, s)
print(" ".join(specs))
print(len(specs), "specs", file=sys.stderr)

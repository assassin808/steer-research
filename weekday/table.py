"""print compact detection tables per run"""
import sys, os, numpy as np, aggregate as A
STATS = [("id", ["flatG", "blob"]), ("distortion", ["flatG", "line2"]), ("cyc_G", ["shuffle", "line2"]), ("cyc_E", ["shuffle", "line2"]),
         ("turn_corr_med", ["line2", "shuffle"]), ("top1", ["line2"]), ("chord2_med", ["line2"]), ("field_rot_med", ["line2", "shuffle"]),
         ("adj", ["shuffle", "line2"])]
for fn in sys.argv[1:]:
    s = A.geom_summary(fn); print(f"== {os.path.basename(fn)} n={s['n']}")
    for w in ["mid", "late"]:
        d = s[f"{w}_detect"]
        print(f"  [{w}] " + " | ".join(f"{st} {d[st+'|data_median']:.2f} " + ",".join(f"{nm}:{d[st+'|'+nm]}" for nm in nl) for st, nl in STATS))

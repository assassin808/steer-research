"""[harness addition] MC1/MC2 (gm_mc.py, unchanged) for confirm-grad specs incl. bridge 'br:' specs -> results_gm/mc_confirm."""
import argparse, os, sys
import pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import bridge as B  # noqa
import gm_mc as M  # noqa
import ng_lib as L  # noqa
orig = M.build
M.build = lambda ctx, spec: B.build_bridge(ctx, spec) if spec.startswith("br:") else orig(ctx, spec)
ap = argparse.ArgumentParser(); ap.add_argument("--specs", nargs="+", required=True); a = ap.parse_args()
out = os.path.join(M.ROOT, "results_gm", "mc_confirm"); os.makedirs(out, exist_ok=True)
ctx = M.Ctx(); M.DF = pd.read_csv(ctx.task._csv_path())
tok, model = L.load_model(torch.bfloat16)
steers = {}
for s in a.specs:
    M.run(ctx, model, tok, steers, s, out)

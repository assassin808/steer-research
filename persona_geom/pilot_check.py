"""Pilot design check (prereg amendment A1, section 7.1). CPU, on RUN=pilot (gen main + judge done).
On TRAIN rows of acts_neu at layers 8, 12, 16 (the steering layer is not chosen yet), per trait:
  - kappa (geom.cell_kappa; power/designcheck.py unit; prior .55), overall and per level;
  - SNR_x: split-half (cross-Gram) SNR of the one-sided placebo-referenced ladder, trace(M) / L / W_eps (descriptive);
  - PROJECTED POWER of the main design: the pilot's own residual cells, questions resampled with replacement to Q_main = 120,
    a 60-degree arc planted at the pilot ladder's split-half positions in the hardest (top residual PC) plane, tested with the
    primary test at the conservative Holm threshold .02/6 (geom.planted_power, Qboot), under the Sigma_w metric M2 (A2). This uses the real noise, cell effects and
    their level dependence, so no SNR-unit conversion from the power curve is needed (review of A1, findings 3, 5, 8).
  - H0 / wording rule (analyze_geom.prompt_profile).
design_ok = wording rule passes AND >= 3 H0-passing traits have median (over the 3 layers) projected power >= .8.
usage: RUN=pilot python pilot_check.py   -> <OUT>/pilot_check.json
"""
import os, sys, numpy as np
sys.argv = sys.argv[:1]
from common import OUT, TRAITS, jdump, log, cells
import geom as G
from analyze_geom import load_meta, prompt_profile

Q_MAIN = int(os.environ.get("Q_MAIN_TRAIN", 120)); NSIM = int(os.environ.get("NSIM_PILOT", 2 if os.environ.get("SMOKE") else 40))
LAYERS = [int(x) for x in os.environ.get("PILOT_LAYERS", "8,12,16").split(",")]

if __name__ == "__main__":
    rows, meta0 = load_meta(); pp = prompt_profile(meta0); jdump(pp, f"{OUT}/prompt_profile.json")
    tr = meta0["split"] == "train"; idx = np.where(tr)[0]; meta = {k: v[tr] for k, v in meta0.items()}
    A = np.load(f"{OUT}/acts_neu.npy", mmap_mode="r"); res = {t: {} for t in TRAITS}; rng = np.random.default_rng(0)
    for L in LAYERS:
        X = G.demean_by(np.asarray(A[idx, L], np.float64), meta["q"])
        for t in TRAITS:
            lad = G.ladder(meta, t); mL = {k: v[lad] for k, v in meta.items()}
            c = G.curve_stats(X[lad], mL, t); kap = G.cell_kappa(X[lad], mL, t, C=len(cells()))
            lc = G.ladder_cells(X[lad], mL, t); maps, _ = G.sigma_w_maps(X, meta, exclude=(t, "placebo"))   # A2: M2, leave-trait-out
            pw = G.planted_power(G.apply_metric(lc["D"], maps["M2"]), lc, rng, .02 / len(TRAITS), 60.0, "noise", nsim=NSIM, Qboot=Q_MAIN)
            res[t][L] = dict(snr_x=float(np.trace(c["_M"]) / len(c["_M"]) / kap["W_eps"]), ord1=c["ord1"], quadperp=c["quadperp"],
                             kappa=kap["kappa"], kappa_by_level=kap["kappa_by_level"], proj_power=pw["power"], proj_power_lo90=pw["lo90"])
            log("pilot", L, t, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[t][L].items() if k != "kappa_by_level"})
    out = dict(Q_main=Q_MAIN, nsim=NSIM, layers=LAYERS, per_trait={})
    npow = 0
    for t in TRAITS:
        med = float(np.median([res[t][L]["proj_power"] for L in LAYERS])); h0 = pp["_H0"][t]["H0"]
        ok = bool(h0 and med >= .8); npow += ok
        out["per_trait"][t] = dict(H0=h0, gain=pp["_H0"][t]["gain"], graded=pp["_H0"][t]["graded"], proj_power_median=med,
                                   powered=ok, by_layer={str(L): res[t][L] for L in LAYERS})
    out.update(kappa_median=float(np.median([res[t][L]["kappa"] for t in TRAITS for L in LAYERS])),
               wording_rule_pass=pp["_wording_rule_pass"], n_powered=int(npow), design_ok=bool(pp["_wording_rule_pass"] and npow >= 3))
    jdump(out, f"{OUT}/pilot_check.json")
    log("pilot_check", {k: out[k] for k in ("kappa_median", "wording_rule_pass", "n_powered", "design_ok")})

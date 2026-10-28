"""Coverage of a rank-stratified percentile bootstrap vs Wald for the RSS ratio estimator."""
import numpy as np, json
from sim import draw, true_p, proxy

def boot_cover(m, r, gamma, design, reps=2000, B=499, seed=7):
    rng = np.random.default_rng(seed)
    p = true_p(gamma)
    W, pi, S = draw(rng, (reps, r, m, m), gamma, 0.10)
    wk = W[..., None] * pi
    if design == "SRS":
        Ws, ws = W[..., 0], wk[..., 0, :]
    else:
        x = proxy(design, W, S) + 1e-9 * rng.random(W.shape)
        idx = np.argsort(x, -1)[:, :, np.arange(m), np.arange(m)]
        Ws = np.take_along_axis(W, idx[..., None], -1)[..., 0]
        ws = np.take_along_axis(wk, idx[..., None, None], -2)[..., 0, :]
    est = ws.sum((1, 2)) / Ws.sum((1, 2))[:, None]
    # resample cycles independently within each rank class (SRS: within each set slot, equivalent to iid)
    bi = rng.integers(0, r, size=(B, reps, r, m))
    Wb = np.take_along_axis(Ws[None], bi, 2)               # (B,reps,r,m)
    wb = np.take_along_axis(ws[None], bi[..., None], 2)
    eb = wb.sum((2, 3)) / Wb.sum((2, 3))[..., None]         # (B,reps,6)
    lo, hi = np.quantile(eb, [0.025, 0.975], axis=0)
    # basic (reverse percentile) interval
    blo, bhi = 2 * est - hi, 2 * est - lo
    return ((lo <= p) & (p <= hi)).mean(0), ((blo <= p) & (p <= bhi)).mean(0)

out = []
for gamma in (1.0,):
    for r in (5, 10, 20):
        for d in ("SRS", "WEIGHT", "RESID"):
            pc, bc = boot_cover(3, r, gamma, d)
            out.append(dict(gamma=gamma, r=r, design=d, perc=pc.tolist(), basic=bc.tolist()))
            print(r, d, np.round(pc, 3), np.round(bc, 3), flush=True)
json.dump(out, open("boot_results.json", "w"), indent=1)

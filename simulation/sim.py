"""
Simulation study: ranked set sampling (RSS) for municipal solid waste composition.

Superpopulation of sampling units (household bags / collection loads):
  log W ~ N(mu, s^2)                       total mass (kg), mean 8 kg, CV ~ 0.66
  z = standardised log W
  eta_k = log(q_k) + gamma*b_k*z + tau*e_k   e_k ~ N(0,1)
  pi = softmax(eta)                          share vector of the unit
  w_k = W * pi_k                             fraction masses
Target: p_k = E[w_k]/E[W]  (mass-weighted population share).

Ranking proxies (one ranking serves all fractions):
  WEIGHT : W (scale reading, exact)
  SCORE  : visual organic score S in {0..5}, S = round(5*(pi_food + delta)), delta~N(0,sv)
  ORGMASS: W*S/5  (visual organic mass)
  RESID  : W*(S/5 - c0), c0 = 0.5 prior guess of food share (estimated food residual)
Estimator: ratio estimator sum(w_k)/sum(W) over the n=m*r sorted units.
SRS comparator: one random unit per set (same n, same draws).
"""
import numpy as np, itertools, json, sys, time

FR = ["Food/organic", "Paper/card", "Plastics", "Glass", "Metals", "Other"]
Q = np.array([0.50, 0.12, 0.13, 0.06, 0.04, 0.15])
B = np.array([0.50, -0.30, -0.40, 0.30, 0.00, 0.00])   # share gradient on log-weight
TAU = 0.6
MW, CVW = 8.0, 0.66
SLOG = np.sqrt(np.log(1 + CVW**2)); MLOG = np.log(MW) - SLOG**2 / 2
C0 = 0.5
PROXIES = ["WEIGHT", "SCORE", "ORGMASS", "RESID"]


def draw(rng, shape, gamma, sv):
    lw = rng.normal(MLOG, SLOG, size=shape)
    z = (lw - MLOG) / SLOG
    eta = np.log(Q) + gamma * B * z[..., None] + TAU * rng.standard_normal(shape + (6,))
    eta -= eta.max(-1, keepdims=True)
    pi = np.exp(eta); pi /= pi.sum(-1, keepdims=True)
    W = np.exp(lw)
    S = np.clip(np.rint(5 * (pi[..., 0] + rng.normal(0, sv, size=shape))), 0, 5)
    return W, pi, S


def true_p(gamma, n=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    W, pi, _ = draw(rng, (n,), gamma, 0.1)
    return (W[:, None] * pi).sum(0) / W.sum()


def proxy(name, W, S):
    if name == "WEIGHT": return W
    if name == "SCORE": return S
    if name == "ORGMASS": return W * S / 5
    if name == "RESID": return W * (S / 5 - C0)


def one_cell(m, r, gamma, sv, reps, rng, p):
    """Returns dict proxy -> (est array reps x 6, var-estimate array) plus SRS."""
    W, pi, S = draw(rng, (reps, r, m, m), gamma, sv)
    wk = W[..., None] * pi
    out = {}
    # SRS: unit 0 of each set (units are iid so this is a simple random sample of size m*r)
    Ws, wks = W[..., 0], wk[..., 0, :]
    out["SRS"] = summarize(Ws.reshape(reps, -1), wks.reshape(reps, -1, 6), None, m, r)
    for name in PROXIES:
        x = proxy(name, W, S) + 1e-9 * rng.random(W.shape)       # random tie-break
        order = np.argsort(x, axis=-1)                             # (reps,r,m,m)
        idx = order[:, :, np.arange(m), np.arange(m)]              # rank i taken from set i
        Wsel = np.take_along_axis(W, idx[..., None], -1)[..., 0]   # (reps,r,m)
        wsel = np.take_along_axis(wk, idx[..., None, None], -2)[..., 0, :]
        out[name] = summarize(Wsel, wsel, "rss", m, r)
    return out


def summarize(Wsel, wsel, kind, m, r):
    reps = Wsel.shape[0]
    Wf = Wsel.reshape(reps, -1); wf = wsel.reshape(reps, -1, 6)
    est = wf.sum(1) / Wf.sum(1)[:, None]
    n = Wf.shape[1]; wbar = Wf.mean(1)
    d = wf - est[:, None, :] * Wf[..., None]
    if kind is None:  # SRS linearised variance
        v = d.var(1, ddof=1) / (n * wbar[:, None] ** 2)
    else:             # RSS: within-rank-class variance across cycles
        dd = d.reshape(reps, r, m, 6)
        v = dd.var(1, ddof=1).sum(1) * r / (n**2 * wbar[:, None] ** 2)
    return est, v


def run(reps=10000, seed=2026):
    rng = np.random.default_rng(seed)
    res = []
    ptrue = {g: true_p(g) for g in (0.0, 1.0)}
    for gamma, sv, m in itertools.product((0.0, 1.0), (0.05, 0.10, 0.20), (2, 3, 4, 5)):
        r = 10
        p = ptrue[gamma]
        out = one_cell(m, r, gamma, sv, reps, rng, p)
        mse_srs = ((out["SRS"][0] - p) ** 2).mean(0)
        for name, (est, v) in out.items():
            mse = ((est - p) ** 2).mean(0)
            bias = (est - p).mean(0)
            se = np.sqrt(v)
            cover = (np.abs(est - p) <= 1.96 * se).mean(0)
            res.append(dict(gamma=gamma, sv=sv, m=m, r=r, design=name,
                            p=p.tolist(), bias=bias.tolist(), mse=mse.tolist(),
                            re=(mse_srs / mse).tolist(),
                            relbias=(bias / p).tolist(), cover=cover.tolist()))
        print(gamma, sv, m, {k: np.round(mse_srs / ((out[k][0] - p) ** 2).mean(0), 2).tolist() for k in PROXIES}, flush=True)
    # coverage vs r for the RESID and WEIGHT designs, m=3
    cov = []
    for gamma, r in itertools.product((0.0, 1.0), (5, 10, 20)):
        p = ptrue[gamma]
        out = one_cell(3, r, gamma, 0.10, reps, rng, p)
        for name, (est, v) in out.items():
            cov.append(dict(gamma=gamma, r=r, m=3, design=name,
                            cover=(np.abs(est - p) <= 1.96 * np.sqrt(v)).mean(0).tolist()))
    json.dump(dict(fractions=FR, ptrue={str(k): v.tolist() for k, v in ptrue.items()},
                   cells=res, coverage=cov), open("sim_results.json", "w"), indent=1)


if __name__ == "__main__":
    t = time.time(); run(int(sys.argv[1]) if len(sys.argv) > 1 else 10000); print("done", time.time() - t)

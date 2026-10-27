"""
rbys: 'Rank before you sort' -- dataset-agnostic resampling engine for ranked set sampling (RSS)
of waste composition audits.

A Population holds, for N fully sorted units, the unit mass W, the fraction masses wk (N x K, organics
in column 0), the pre-sort scale reading, and any proxies recorded without sorting (household size,
bulk density, ...).  The visual organic score is emulated as S = round{5(pi_org + delta)} in 0..5.

Audits are drawn WITH replacement from the fully sorted units, i.e. the units are treated as an
empirical superpopulation (the audited city), so the design does not exhaust a small population.
A without-replacement variant is kept for comparison with the first submission.
"""
import numpy as np
from scipy import stats

D_M = {2: 0.318310, 3: 0.477465, 4: 0.573917, 5: 0.638893}   # (1/m) sum alpha_{i:m}^2
DESIGNS = ((2, 15), (3, 10), (4, 8))                           # 30, 30, 32 sorted units


class Population:
    def __init__(self, name, fractions, W, wk, scale=None, recorded=None, strata=None):
        self.name, self.F = name, list(fractions)
        self.W = np.asarray(W, float)
        self.wk = np.asarray(wk, float)
        self.N, self.K = self.wk.shape
        self.scale = self.W.copy() if scale is None else np.asarray(scale, float)
        self.recorded = recorded or {}                         # name -> vector (NaN allowed)
        self.strata = None if strata is None else np.asarray(strata)
        self.p = self.wk.sum(0) / self.W.sum()
        self.pi_org = self.wk[:, 0] / self.W

    # ---------------------------------------------------------------- proxies
    levels = 5                                                  # visual score scale 0..levels

    def proxies(self, rng, idx, sv=0.10, c0=0.5):
        """Proxy values for the units indexed by idx (any shape); fresh scoring error per inspection."""
        L = self.levels
        out = {"WEIGHT": self.scale[idx]}
        for k, v in self.recorded.items():
            v = np.where(np.isnan(v), np.nanmedian(v), v)          # missing -> ordered at random (ties)
            out[k] = v[idx]
        S = np.clip(np.rint(L * (self.pi_org[idx] + rng.normal(0, sv, idx.shape))), 0, L)
        out["SCORE"] = S
        out["ORGMASS"] = self.scale[idx] * S / L
        out["RESID"] = self.scale[idx] * (S / L - c0)
        return out

    def proxy_names(self):
        return ["WEIGHT", *self.recorded.keys(), "SCORE", "ORGMASS", "RESID"]

    # ---------------------------------------------------------------- summaries
    def describe(self):
        sh = self.wk / self.W[:, None]
        return dict(
            name=self.name, N=self.N, fractions=self.F, total_kg=float(self.W.sum()), p=self.p.tolist(),
            unit_mass=dict(mean=float(self.W.mean()), median=float(np.median(self.W)),
                           cv=float(self.W.std(ddof=1) / self.W.mean()), min=float(self.W.min()), max=float(self.W.max())),
            zero_freq=(self.wk == 0).mean(0).tolist(),
            share_weight_spearman=[float(stats.spearmanr(self.W, sh[:, j])[0]) for j in range(self.K)],
            share_weight_p=[float(stats.spearmanr(self.W, sh[:, j])[1]) for j in range(self.K)],
            scale_vs_sum=float(stats.spearmanr(self.scale, self.W)[0]),
            strata=None if self.strata is None else {str(k): int(v) for k, v in zip(*np.unique(self.strata, return_counts=True))},
        )


# -------------------------------------------------------------------- core resampling
def _rank_select(x, rng):
    """x: (..., r, m, m) proxy values -> positions (r, m) of the unit ranked i in set i."""
    m = x.shape[-1]
    o = np.argsort(x + 1e-9 * rng.random(x.shape), -1)
    return o[..., np.arange(m), np.arange(m)]                 # (..., r, m)


def _ratio(wk, W):
    return wk.sum((-3, -2)) / W.sum((-2, -1))[..., None]


def simulate(pop, m, r, reps, rng, sv=0.10, c0=0.5, replace=True, chunk=4000, keep_est=False):
    """Emulate `reps` audits of design (m, r). Returns dict with RE (+MC s.e.), PW-RE, bias, coverage."""
    names = ["SRS", *pop.proxy_names()]
    se = {k: [] for k in names}; cov = {k: [] for k in names}; est_keep = {k: [] for k in names}
    done = 0
    while done < reps:
        b = min(chunk, reps - done); done += b
        if replace:
            idx = rng.integers(0, pop.N, size=(b, r, m, m))
        else:
            idx = np.argsort(rng.random((b, pop.N)), 1)[:, : m * m * r].reshape(b, r, m, m)
        px = pop.proxies(rng, idx, sv, c0)
        sel = {"SRS": idx[..., 0]}
        for k, x in px.items():
            pos = _rank_select(x, rng)
            sel[k] = np.take_along_axis(idx, pos[..., None], -1)[..., 0]
        for k, s in sel.items():
            W = pop.W[s]; w = pop.wk[s]                            # (b,r,m), (b,r,m,K)
            e = w.sum((1, 2)) / W.sum((1, 2))[:, None]
            d = w - e[:, None, None, :] * W[..., None]
            n = m * r
            if k == "SRS":
                v = d.reshape(b, n, -1).var(1, ddof=1) / (n * W.mean((1, 2))[:, None] ** 2); df = n - 1
            else:
                v = r * d.var(1, ddof=1).sum(1) / (n ** 2 * W.mean((1, 2))[:, None] ** 2); df = m * (r - 1)
            se[k].append((e - pop.p) ** 2)
            cov[k].append(np.abs(e - pop.p) <= stats.t.ppf(0.975, df) * np.sqrt(v))
            if keep_est: est_keep[k].append(e)
    se = {k: np.concatenate(v) for k, v in se.items()}
    cov = {k: np.concatenate(v).mean(0) for k, v in cov.items()}
    mse = {k: v.mean(0) for k, v in se.items()}
    w_rel = 1 / np.maximum(pop.p, 1e-12) ** 2
    out = dict(m=m, r=r, n=m * r, inspected=m * m * r, reps=reps, replace=replace, sv=sv, c0=c0,
               re={}, re_se={}, pwre={}, pwre_se={}, relbias={}, cover={k: v.tolist() for k, v in cov.items()},
               rmse={k: np.sqrt(v).tolist() for k, v in mse.items()})
    a = se["SRS"]
    for k in names:
        bb = se[k]
        re = a.mean(0) / bb.mean(0)
        # delta-method Monte Carlo s.e. of a ratio of paired means
        va, vb = a.var(0, ddof=1), bb.var(0, ddof=1)
        cab = ((a - a.mean(0)) * (bb - bb.mean(0))).sum(0) / (reps - 1)
        re_se = re * np.sqrt((va / a.mean(0) ** 2 + vb / bb.mean(0) ** 2 - 2 * cab / (a.mean(0) * bb.mean(0))) / reps)
        out["re"][k] = re.tolist(); out["re_se"][k] = np.nan_to_num(re_se).tolist()
        A, B = (a * w_rel).sum(1), (bb * w_rel).sum(1)
        pw = A.mean() / B.mean()
        cAB = np.cov(A, B)
        out["pwre"][k] = float(pw)
        out["pwre_se"][k] = float(pw * np.sqrt((cAB[0, 0] / A.mean() ** 2 + cAB[1, 1] / B.mean() ** 2
                                                - 2 * cAB[0, 1] / (A.mean() * B.mean())) / reps))
    if keep_est:
        out["_est"] = {k: np.concatenate(v) for k, v in est_keep.items()}
    return out


def relbias(pop, m, r, reps, rng, **kw):
    """Relative bias (needs the estimates themselves)."""
    o = simulate(pop, m, r, reps, rng, keep_est=True, **kw)
    return {k: ((v.mean(0) - pop.p) / pop.p).tolist() for k, v in o["_est"].items()}


# -------------------------------------------------------------------- planning rule
def nscore(x):
    return stats.norm.ppf(stats.rankdata(x) / (len(x) + 1))


def population_rho(pop, rng, reps=200, sv=0.10, c0=0.5):
    d = pop.wk - np.outer(pop.W, pop.p)
    acc = {}
    all_idx = np.arange(pop.N)
    for _ in range(reps):
        for k, x in pop.proxies(rng, all_idx, sv, c0).items():
            z = nscore(x + 1e-9 * rng.random(pop.N))
            acc.setdefault(k, []).append([np.corrcoef(z, d[:, j])[0, 1] if d[:, j].std() > 0 else 0.0
                                          for j in range(pop.K)])
    return {k: np.mean(v, 0).tolist() for k, v in acc.items()}


def predicted_re(rho, m):
    rho = np.asarray(rho)
    return (1 / (1 - rho ** 2 * D_M[m])).tolist()


def adjusted_rho2(rho, n0):
    """Small-sample adjustment of a squared correlation (as for adjusted R^2), floored at 0."""
    rho2 = np.nan_to_num(np.asarray(rho)) ** 2
    return np.maximum(0.0, 1 - (1 - rho2) * (n0 - 1) / (n0 - 2))


def pilot_study(pop, re_m3, pwre_m3, rng, reps=2000, n0=20, sv=0.10, c0=0.5, cost_ratio=0.03, m=3):
    """Simulated pilots of n0 sorted units: estimate rho, predict RE (set size m), choose a proxy or SRS.

    rule 'naive'   : raw rho^2, pick the best predicted criterion (first submission).
    rule 'adjusted': adjusted rho^2 and rank only if the predicted saving beats the inspection cost,
                     i.e. criterion x (1 + m c_r/c_s) < SRS criterion (Eq. cost model).
    Realised performance is reported both as RE and as the total-cost ratio C_x / C_SRS at equal precision.
    """
    infl = 1 + m * cost_ratio
    choices = {(rule, pol): [] for rule in ("naive", "adjusted") for pol in ("organics", "equal")}
    for _ in range(reps):
        s = rng.choice(pop.N, n0, replace=False)
        W = pop.W[s]; w = pop.wk[s]
        ph = w.sum(0) / W.sum(); d = w - np.outer(W, ph)
        cv2 = d.var(0, ddof=1) / (n0 * W.mean() ** 2) / np.maximum(ph, 1e-3) ** 2
        rhos = {}
        for k, x in pop.proxies(rng, s, sv, c0).items():
            z = nscore(x + 1e-9 * rng.random(n0))
            rhos[k] = np.array([np.corrcoef(z, d[:, j])[0, 1] if d[:, j].std() > 0 else 0 for j in range(pop.K)])
        for rule in ("naive", "adjusted"):
            if rule == "naive":
                pred = {k: 1 / (1 - np.nan_to_num(r) ** 2 * D_M[m]) for k, r in rhos.items()}; f = 1.0
            else:
                pred = {k: 1 / (1 - adjusted_rho2(r, n0) * D_M[m]) for k, r in rhos.items()}; f = infl
            c_org = {k: f / v[0] for k, v in pred.items()}; c_org["SRS"] = 1.0
            c_eq = {k: f * (cv2 / v).sum() for k, v in pred.items()}; c_eq["SRS"] = cv2.sum()
            choices[(rule, "organics")].append(min(c_org, key=c_org.get))
            choices[(rule, "equal")].append(min(c_eq, key=c_eq.get))
    out = {}
    for (rule, pol), ch in choices.items():
        vals, cnt = np.unique(ch, return_counts=True)
        re_of = (lambda c: re_m3[c][0]) if pol == "organics" else (lambda c: pwre_m3[c])
        realised = np.array([re_of(c) for c in ch])
        cost = np.array([1.0 if c == "SRS" else infl / re_of(c) for c in ch])
        best = max(re_of(c) for c in re_m3)
        best_cost = min([1.0] + [infl / re_of(c) for c in re_m3 if c != "SRS"])
        out[f"{rule}:{pol}"] = dict(freq={v: int(c) for v, c in zip(vals, cnt)},
                                    mean_realised=float(realised.mean()), best_attainable=float(best),
                                    mean_cost_ratio=float(cost.mean()), best_cost_ratio=float(best_cost),
                                    p_cost_gt1=float((cost > 1).mean()))
    return out


# -------------------------------------------------------------------- population bootstrap
def population_bootstrap(pop, rng, B=200, reps=2000, m=3, r=10, sv=0.10, c0=0.5):
    """Uncertainty in RE due to the finite number of sorted households: resample households,
    rebuild the population, and re-run the audit emulation."""
    res = {k: [] for k in ["SRS", *pop.proxy_names()]}
    pw = {k: [] for k in res}
    for _ in range(B):
        j = rng.integers(0, pop.N, pop.N)
        rec = {k: v[j] for k, v in pop.recorded.items()}
        bp = Population(pop.name, pop.F, pop.W[j], pop.wk[j], pop.scale[j], rec); bp.levels = pop.levels
        o = simulate(bp, m, r, reps, rng, sv, c0)
        for k in res:
            res[k].append(o["re"][k]); pw[k].append(o["pwre"][k])
    q = lambda a: np.nanpercentile(np.asarray(a, float), [2.5, 50, 97.5], axis=0).tolist()
    return dict(B=B, reps=reps, m=m, r=r, re_q={k: q(v) for k, v in res.items()},
                pwre_q={k: q(v) for k, v in pw.items()},
                prob_re_gt1_org={k: float((np.asarray(v)[:, 0] > 1).mean()) for k, v in res.items()})


# -------------------------------------------------------------------- stratified RSS
def simulate_stratified(pop, m, r_h, reps, rng, sv=0.10, c0=0.5):
    """RSS within each stratum (r_h cycles in stratum h), combined ratio estimator with stratum weights
    N_h (empirical stratum sizes).  Compared with stratified SRS of the same allocation."""
    strata = np.unique(pop.strata)
    Nh = np.array([(pop.strata == h).sum() for h in strata], float); Wh = Nh / Nh.sum()
    names = ["SRS", *pop.proxy_names()]
    tot_w = {k: 0 for k in names}; tot_W = {k: 0 for k in names}
    for h, wh, rh in zip(strata, Wh, r_h):
        units = np.flatnonzero(pop.strata == h)
        idx = units[rng.integers(0, len(units), size=(reps, rh, m, m))]
        px = pop.proxies(rng, idx, sv, c0)
        sel = {"SRS": idx[..., 0]}
        for k, x in px.items():
            pos = _rank_select(x, rng)
            sel[k] = np.take_along_axis(idx, pos[..., None], -1)[..., 0]
        for k, s in sel.items():
            nh = m * rh
            tot_w[k] = tot_w[k] + wh * pop.wk[s].sum((1, 2)) / nh
            tot_W[k] = tot_W[k] + wh * pop.W[s].sum((1, 2)) / nh
    est = {k: tot_w[k] / tot_W[k][:, None] for k in names}
    # target: stratum-weighted population ratio
    mu_w = sum(wh * pop.wk[pop.strata == h].mean(0) for h, wh in zip(strata, Wh))
    mu_W = sum(wh * pop.W[pop.strata == h].mean() for h, wh in zip(strata, Wh))
    p = mu_w / mu_W
    mse = {k: ((v - p) ** 2).mean(0) for k, v in est.items()}
    w_rel = 1 / np.maximum(p, 1e-12) ** 2
    return dict(m=m, r_h=list(map(int, r_h)), re={k: (mse["SRS"] / mse[k]).tolist() for k in names},
                pwre={k: float((mse["SRS"] * w_rel).sum() / (mse[k] * w_rel).sum()) for k in names})

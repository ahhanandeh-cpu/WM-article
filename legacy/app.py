"""
Application: resampling study on the Cefas CLiP Solomon Islands household waste audit (2018).
Population: 174 two-day household samples from Guadalcanal (80) and Malaita (Arabella 37,
Kilusakwalo 33, Ambu 24). Each was fully sorted, so true shares are known.
"""
import numpy as np, pandas as pd, json, sys
from scipy import stats

F = ["Organics", "Paper", "Plastics", "Metals", "Glass", "Hygiene", "Other"]
U = pd.read_csv("data/units.csv", index_col=0)
P = U[U.index.str.match(r"^(AH|AR|R)\d") & (U.mult == 2)].copy()
I = pd.read_csv("data/interviews.csv").rename(columns={"Household/Commercial/Ship_ID": "ID"}).set_index("ID")
P = P.join(I[["Number_of_people_in_household", "Grocery_expenses_US"]])
N = len(P)
Wt = P["sum"].to_numpy()                       # true unit mass (sum of sorted fractions)
wk = P[F].to_numpy()                           # fraction masses
scale = np.where(P.total.to_numpy() > 0, P.total.to_numpy(), Wt)   # recorded scale weight before sorting
hh = P["Number_of_people_in_household"].to_numpy()
pi_org = wk[:, 0] / Wt
p = wk.sum(0) / Wt.sum()
C0 = 0.5


def proxies(rng, sv=0.10):
    """Return dict of proxy vectors for the whole population (fresh scoring noise each call)."""
    S = np.clip(np.rint(5 * (pi_org + rng.normal(0, sv, N))), 0, 5)
    h = np.where(np.isnan(hh), np.nanmedian(hh), hh)
    return {"WEIGHT": scale, "HHSIZE": h, "SCORE": S, "ORGMASS": scale * S / 5,
            "RESID": scale * (S / 5 - C0)}


def one_audit(rng, m, r, sv=0.10):
    pr = proxies(rng, sv)
    idx = rng.permutation(N)[: m * m * r].reshape(r, m, m)     # without replacement
    out = {}
    srs = idx[:, :, 0].ravel()
    out["SRS"] = (srs, None)
    for k, x in pr.items():
        xx = x[idx] + 1e-9 * rng.random(idx.shape)
        o = np.argsort(xx, -1)
        sel = np.take_along_axis(idx, o, -1)[:, np.arange(m), np.arange(m)]   # (r,m) rank i from set i
        out[k] = (sel.ravel(), sel)
    return out


def estimate(sel):
    return wk[sel].sum(0) / Wt[sel].sum()


def varhat(sel2d, m, r):
    W = Wt[sel2d]; w = wk[sel2d]; ph = w.sum((0, 1)) / W.sum()
    d = w - ph * W[..., None]
    n = m * r
    return r * d.var(0, ddof=1).sum(0) / (n**2 * W.mean() ** 2)


def run(reps=20000, seed=10):
    rng = np.random.default_rng(seed)
    res = {}
    for m, r in ((2, 15), (3, 10), (4, 8)):
        est = {k: [] for k in ["SRS", "WEIGHT", "HHSIZE", "SCORE", "ORGMASS", "RESID"]}
        cov = {k: [] for k in est}
        for _ in range(reps):
            a = one_audit(rng, m, r)
            for k, (sel, sel2) in a.items():
                e = estimate(sel); est[k].append(e)
                if sel2 is None:
                    W = Wt[sel]; d = wk[sel] - e * W[:, None]
                    v = d.var(0, ddof=1) / (len(sel) * W.mean() ** 2); df = len(sel) - 1
                else:
                    v = varhat(sel2, m, r); df = m * (r - 1)
                cov[k].append(np.abs(e - p) <= stats.t.ppf(0.975, df) * np.sqrt(v))
        mse = {k: ((np.array(v) - p) ** 2).mean(0) for k, v in est.items()}
        res[f"{m},{r}"] = dict(
            m=m, r=r, n=m * r, inspected=m * m * r,
            re={k: (mse["SRS"] / mse[k]).tolist() for k in est},
            pwre={k: float((mse["SRS"] / p**2).sum() / (mse[k] / p**2).sum()) for k in est},
            relbias={k: ((np.array(v).mean(0) - p) / p).tolist() for k, v in est.items()},
            cover={k: np.array(c).mean(0).tolist() for k, c in cov.items()},
            rmse={k: np.sqrt(mse[k]).tolist() for k in est})
        print(m, r, {k: np.round(v, 2).tolist() for k, v in res[f"{m},{r}"]["re"].items()}, flush=True)
        if (m, r) == (3, 10):
            np.save("app_est_m3.npy", {k: np.array(v) for k, v in est.items()}, allow_pickle=True)
    return res


def nscore(x):
    return stats.norm.ppf(stats.rankdata(x) / (len(x) + 1))


def population_rho(rng, reps=200):
    """Normal-score correlation of each proxy with each residual in the full population (avg over score noise)."""
    d = wk - np.outer(Wt, p)
    acc = {}
    for _ in range(reps):
        for k, x in proxies(rng).items():
            z = nscore(x + 1e-9 * rng.random(N))
            acc.setdefault(k, []).append([np.corrcoef(z, d[:, j])[0, 1] for j in range(len(F))])
    return {k: np.mean(v, 0).tolist() for k, v in acc.items()}


def pilot_study(res, reps=2000, n0=20, seed=5):
    """Simulate a 20-household pilot: estimate rho, predict RE, pick a proxy; score the choice."""
    D = {2: 0.3183, 3: 0.4775, 4: 0.5739}
    rng = np.random.default_rng(seed)
    choices = {"organics": [], "equal": []}
    for _ in range(reps):
        s = rng.choice(N, n0, replace=False)
        ph = wk[s].sum(0) / Wt[s].sum(); d = wk[s] - np.outer(Wt[s], ph)
        cv2 = d.var(0, ddof=1) / (n0 * Wt[s].mean() ** 2) / np.maximum(ph, 1e-3) ** 2
        pr = proxies(rng)
        pred = {}
        for k, x in pr.items():
            z = nscore(x[s] + 1e-9 * rng.random(n0))
            rho = np.array([np.corrcoef(z, d[:, j])[0, 1] if d[:, j].std() > 0 else 0 for j in range(len(F))])
            pred[k] = 1 / (1 - rho**2 * D[3])
        crit_org = {k: 1 / v[0] for k, v in pred.items()}
        crit_eq = {k: (cv2 / v).sum() for k, v in pred.items()}
        crit_org["SRS"] = 1.0; crit_eq["SRS"] = cv2.sum()
        choices["organics"].append(min(crit_org, key=crit_org.get))
        choices["equal"].append(min(crit_eq, key=crit_eq.get))
    out = {}
    re3 = res["3,10"]
    for pol, ch in choices.items():
        vals, cnt = np.unique(ch, return_counts=True)
        realised = [re3["re"][c][0] if pol == "organics" else re3["pwre"][c] for c in ch]
        out[pol] = dict(freq={v: int(c) for v, c in zip(vals, cnt)}, mean_realised_re=float(np.mean(realised)))
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = np.random.default_rng(99)
    sh = wk / Wt[:, None]
    summary = dict(
        N=N, sites=P.site.value_counts().to_dict(), total_kg=float(Wt.sum()),
        p=p.tolist(), fractions=F,
        unit_mass=dict(mean=float(Wt.mean()), median=float(np.median(Wt)), cv=float(Wt.std(ddof=1) / Wt.mean()),
                       min=float(Wt.min()), max=float(Wt.max())),
        zero_freq=(wk == 0).mean(0).tolist(),
        share_cv=(sh.std(0, ddof=1) / sh.mean(0)).tolist(),
        mean_unit_share=sh.mean(0).tolist(),
        scale_vs_sum=dict(spearman=float(stats.spearmanr(scale, Wt)[0]),
                          median_abs_diff=float(np.median(np.abs(scale - Wt)))),
        share_weight_spearman=[float(stats.spearmanr(Wt, sh[:, j])[0]) for j in range(len(F))],
        share_weight_p=[float(stats.spearmanr(Wt, sh[:, j])[1]) for j in range(len(F))],
        hh_missing=int(np.isnan(hh).sum()),
        site_comp={s: (P.loc[P.site == s, F].sum() / P.loc[P.site == s, "sum"].sum()).tolist() for s in P.site.unique()},
    )
    summary["rho_pop"] = population_rho(rng)
    res = run(reps)
    summary["resampling"] = res
    summary["pilot"] = pilot_study(res)
    print(json.dumps(summary["pilot"], indent=1))
    json.dump(summary, open("app_results.json", "w"), indent=1)

#!/usr/bin/env python3
"""
rbys_tool -- field tool for 'rank before you sort' composition audits.

  plan      Read a pilot of fully sorted units, estimate the normal-score correlation between each
            candidate proxy and each fraction residual, predict the relative efficiency (RE) of ranking,
            and recommend a proxy (or simple random sampling) for a given priority and cost ratio.
  sheet     Print a randomised recording sheet (CSV) for r cycles of set size m.
  estimate  Read a completed audit (one row per sorted unit with its cycle and rank) and report the
            composition estimate with t-intervals (Eqs. 2 and 5 of the paper).

Examples
  python3 rbys_tool.py plan     examples/pilot.csv --mass W --fractions Organics,Paper,Plastics,Metals,Glass,Other \
                                --proxies W,S --score S --levels 5 --m 3 --cost-ratio 0.03 --priority Organics=1
  python3 rbys_tool.py sheet    --m 3 --r 10 --seed 1 > sheet.csv
  python3 rbys_tool.py estimate examples/audit.csv --mass W --fractions Organics,Paper,Plastics,Metals,Glass,Other
"""
import argparse, sys
import numpy as np, pandas as pd
from scipy import stats

D_M = {2: 0.318310, 3: 0.477465, 4: 0.573917, 5: 0.638893}


def nscore(x):
    return stats.norm.ppf(stats.rankdata(x) / (len(x) + 1))


def adj_rho2(rho, n):
    return max(0.0, 1 - (1 - rho ** 2) * (n - 1) / (n - 2))


def cmd_plan(a):
    df = pd.read_csv(a.pilot)
    F = a.fractions.split(","); W = df[a.mass].to_numpy(float); w = df[F].to_numpy(float); n = len(df)
    ph = w.sum(0) / W.sum(); d = w - np.outer(W, ph)
    cv2 = d.var(0, ddof=1) / (n * W.mean() ** 2) / np.maximum(ph, 1e-3) ** 2
    prox = {}
    for p in a.proxies.split(","):
        prox[p] = df[p].to_numpy(float)
    if a.score:
        S = df[a.score].to_numpy(float); c0 = ph[0] if a.c0 is None else a.c0
        prox["organic mass W*S/L"] = W * S / a.levels
        prox[f"estimated residual W*(S/L-{c0:.2f})"] = W * (S / a.levels - c0)
    pri = {f: 0.0 for f in F}
    for kv in a.priority.split(","):
        k, v = kv.split("="); pri[k] = float(v)
    aw = np.array([pri[f] for f in F])
    infl = 1 + a.m * a.cost_ratio
    print(f"Pilot: {n} sorted units; estimated shares: " + ", ".join(f"{f} {100*p:.1f}%" for f, p in zip(F, ph)))
    print(f"Set size m={a.m}; break-even RE = 1 + m*c_r/c_s = {infl:.3f}\n")
    rows = []
    for k, x in prox.items():
        z = nscore(x + 1e-9 * np.random.default_rng(0).random(n))
        rho = np.array([np.corrcoef(z, d[:, j])[0, 1] if d[:, j].std() > 0 else 0.0 for j in range(len(F))])
        re = np.array([1 / (1 - adj_rho2(r, n) * D_M[a.m]) for r in np.nan_to_num(rho)])
        crit = infl * (aw * cv2 / re).sum()
        rows.append((k, rho, re, crit))
    srs = (aw * cv2).sum()
    hdr = "proxy".ljust(36) + "".join(f"{f[:9]:>10}" for f in F) + "   cost-adj. criterion"
    print("normal-score correlation rho (top) and predicted RE after small-sample adjustment (bottom)")
    print(hdr)
    for k, rho, re, crit in rows:
        print(k[:36].ljust(36) + "".join(f"{r:>+10.2f}" for r in rho))
        print("".ljust(36) + "".join(f"{r:>10.2f}" for r in re) + f"   {crit/srs:8.3f}")
    print("SRS (no ranking)".ljust(36) + " " * 10 * len(F) + f"   {1:8.3f}")
    best = min(rows, key=lambda t: t[3])
    if 0.97 * srs <= best[3] < srs:
        print(f"\nRecommendation: marginal. The best proxy ('{best[0]}') is predicted to save under 3% of total "
              f"cost; simple random sampling is equally good unless sorter exposure is the main concern "
              f"(sorted units needed: {100/np.average(best[2], weights=np.maximum(aw,1e-12)):.0f}% of SRS).")
    elif best[3] < srs:
        print(f"\nRecommendation: rank by '{best[0]}' with m={a.m}. Predicted total-cost ratio at equal "
              f"precision: {best[3]/srs:.2f}; sorted units needed: {100/np.average(best[2], weights=np.maximum(aw,1e-12)):.0f}% "
              f"of an SRS audit (priority-weighted).")
    else:
        print("\nRecommendation: use simple random sampling; no proxy is predicted to pay for its inspection cost.")
    if n < 15:
        print("Warning: pilots below 15 units give very noisy correlations.")


def cmd_sheet(a):
    rng = np.random.default_rng(a.seed)
    rows = []
    for c in range(1, a.r + 1):
        for s in range(1, a.m + 1):
            for u in range(1, a.m + 1):
                rows.append(dict(cycle=c, set=s, slot=u, unit_id="", photo="", W="", S="", proxy="", rank_in_set="",
                                 keep=f"keep if rank {s}", tiebreak=int(rng.integers(0, 1000))))
    pd.DataFrame(rows).to_csv(sys.stdout, index=False)


def cmd_estimate(a):
    df = pd.read_csv(a.audit)
    F = a.fractions.split(","); m = df[a.rank].nunique(); r = df[a.cycle].nunique(); n = len(df)
    W = df[a.mass].to_numpy(float); w = df[F].to_numpy(float)
    ph = w.sum(0) / W.sum()
    d = w - np.outer(W, ph)
    s2 = pd.DataFrame(d, columns=F).groupby(df[a.rank].to_numpy()).var(ddof=1).sum(axis=0).to_numpy()
    v = r * s2 / (n ** 2 * W.mean() ** 2)
    t = stats.t.ppf(0.975, m * (r - 1))
    print(f"RSS audit: m={m} ranks, r={r} cycles, n={n} sorted units, mean unit mass {W.mean():.2f}")
    print(f"{'fraction':<14}{'share %':>9}{'s.e. %':>9}{'95% CI %':>20}{'units with none':>17}")
    for j, f in enumerate(F):
        lo, hi = ph[j] - t * np.sqrt(v[j]), ph[j] + t * np.sqrt(v[j])
        z = (w[:, j] == 0).mean()
        flag = "  (sporadic: interval unreliable)" if z > 0.5 else ""
        print(f"{f:<14}{100*ph[j]:>9.1f}{100*np.sqrt(v[j]):>9.2f}{f'[{100*max(lo,0):.1f}, {100*hi:.1f}]':>20}{100*z:>16.0f}%{flag}")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sp = ap.add_subparsers(dest="cmd", required=True)
p = sp.add_parser("plan"); p.add_argument("pilot"); p.add_argument("--mass", required=True)
p.add_argument("--fractions", required=True, help="comma list; put the scored fraction (organics) first")
p.add_argument("--proxies", required=True, help="comma list of recorded proxy columns, e.g. W,S,household_size")
p.add_argument("--score", help="column holding the visual score (enables composite proxies)")
p.add_argument("--levels", type=float, default=5); p.add_argument("--c0", type=float)
p.add_argument("--m", type=int, default=3, choices=[2, 3, 4, 5]); p.add_argument("--cost-ratio", type=float, default=0.03)
p.add_argument("--priority", default="Organics=1", help="weights a_k, e.g. Organics=1,Metals=0.5")
p.set_defaults(f=cmd_plan)
p = sp.add_parser("sheet"); p.add_argument("--m", type=int, default=3); p.add_argument("--r", type=int, default=10)
p.add_argument("--seed", type=int, default=1); p.set_defaults(f=cmd_sheet)
p = sp.add_parser("estimate"); p.add_argument("audit"); p.add_argument("--mass", required=True)
p.add_argument("--fractions", required=True); p.add_argument("--cycle", default="cycle"); p.add_argument("--rank", default="rank")
p.set_defaults(f=cmd_estimate)

if __name__ == "__main__":
    args = ap.parse_args(); args.f(args)

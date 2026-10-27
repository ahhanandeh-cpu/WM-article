"""Run the full resampling study on every available fully sorted household audit.

usage:  python3 run_all.py [dataset ...] [--reps 20000] [--boot 200]
Writes results/<dataset>.json.
"""
import argparse, json, os, time
import numpy as np
import rbys, datasets

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--reps", type=int, default=20000)
ap.add_argument("--boot", type=int, default=200)
ap.add_argument("--boot-reps", type=int, default=2000)
a = ap.parse_args()
names = a.names or list(datasets.LOADERS)
os.makedirs("results", exist_ok=True)


def c0_of(pop):
    """Prior guess of the organic share as a pilot would supply it: the share rounded to 0.1."""
    return round(float(pop.p[0]), 1)


for name in names:
    t0 = time.time()
    seed = sum(map(ord, name))
    rng = np.random.default_rng(seed)
    pop = datasets.LOADERS[name]()
    c0 = c0_of(pop)
    R = dict(describe=pop.describe(), c0=c0, sv=0.10, levels=pop.levels)
    sh = pop.wk[:, 0] / pop.W
    R["describe"]["org_share_sd"] = float(sh.std(ddof=1))
    R["describe"]["proxies"] = pop.proxy_names()

    # planning rule with population correlations
    R["rho_pop"] = rbys.population_rho(pop, rng, 200, 0.10, c0)
    R["pred"] = {f"{m}": {k: rbys.predicted_re(v, m) for k, v in R["rho_pop"].items()} for m in (2, 3, 4)}

    # main resampling study (superpopulation, with replacement)
    R["main"] = {}
    for m, r in rbys.DESIGNS:
        o = rbys.simulate(pop, m, r, a.reps, rng, 0.10, c0)
        o["relbias"] = rbys.relbias(pop, m, r, min(a.reps, 10000), rng, sv=0.10, c0=c0)
        R["main"][f"{m},{r}"] = o
        print(name, m, r, {k: round(v[0], 2) for k, v in o["re"].items()}, flush=True)

    # without replacement, where the design leaves at least a quarter of the population uninspected
    R["noreplace"] = {}
    for m, r in rbys.DESIGNS:
        if m * m * r <= 0.75 * pop.N:
            o = rbys.simulate(pop, m, r, a.reps, rng, 0.10, c0, replace=False)
            R["noreplace"][f"{m},{r}"] = {k: o[k] for k in ("re", "re_se", "pwre", "inspected")}

    # first-submission setting c0 = 0.5
    o = rbys.simulate(pop, 3, 10, a.reps // 2, rng, 0.10, 0.5)
    R["c0_05"] = {k: o[k] for k in ("re", "pwre")}

    # sensitivity: scoring error x score resolution x prior guess (m = 3)
    R["sens"] = []
    for L in (5, 10):
        pop.levels = L
        for sv in (0.05, 0.10, 0.20, 0.30):
            for cc in sorted({c0, round(max(0.1, c0 - 0.2), 1), round(min(0.9, c0 + 0.2), 1)}):
                o = rbys.simulate(pop, 3, 10, 10000, rng, sv, cc)
                R["sens"].append(dict(levels=L, sv=sv, c0=cc, re={k: o["re"][k] for k in ("SCORE", "ORGMASS", "RESID")},
                                      pwre={k: o["pwre"][k] for k in ("SCORE", "ORGMASS", "RESID")}))
    pop.levels = 5

    # set sizes 2..5 at constant n = 30 (approximately) for the efficiency curve
    R["curve"] = {}
    for m in (2, 3, 4, 5):
        r = int(round(30 / m))
        o = rbys.simulate(pop, m, r, 10000, rng, 0.10, c0)
        R["curve"][str(m)] = {k: o["re"][k] for k in o["re"]}

    # practitioner pilots (n0 = 20)
    m3 = R["main"]["3,10"]
    R["pilot"] = rbys.pilot_study(pop, m3["re"], m3["pwre"], rng, 2000, 20, 0.10, c0)

    # uncertainty due to the finite number of sorted households
    if a.boot:
        R["popboot"] = rbys.population_bootstrap(pop, rng, a.boot, a.boot_reps, 3, 10, 0.10, c0)

    # stratified RSS within sites / income areas (r_h proportional to stratum size, total ~30 sorted)
    if pop.strata is not None:
        st, cnt = np.unique(pop.strata, return_counts=True)
        r_h = np.maximum(2, np.rint(10 * cnt / cnt.sum())).astype(int)
        R["stratified"] = dict(strata=st.tolist(), **rbys.simulate_stratified(pop, 3, r_h, 10000, rng, 0.10, c0))

    R["seconds"] = time.time() - t0
    json.dump(R, open(f"results/{name}.json", "w"), indent=1)
    print(name, "done in", round(R["seconds"]), "s", flush=True)

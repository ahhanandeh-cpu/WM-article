"""Figures for the multi-dataset validation (figures/*.pdf)."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datasets as DS
from make_tables import R, LABEL

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
                     "savefig.dpi": 600, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
COL = {"solomon": "#2a78d6", "vanuatu": "#eb6834", "belize": "#1baf7a", "kampala": "#6250d6"}
MK = {"solomon": "o", "vanuatu": "s", "belize": "D", "kampala": "^"}
PCOL = {"WEIGHT": "#8a8984", "SCORE": "#2a78d6", "RESID": "#eb6834"}
PLAB = {"WEIGHT": "Net weight", "SCORE": "Visual score", "RESID": "Estimated residual"}
os.makedirs("figures", exist_ok=True)
LOAD = {"solomon": DS.load_solomon, "kampala": DS.load_kampala}
LOAD.update({k: getattr(DS, f"load_{k}") for k in ("vanuatu", "belize") if hasattr(DS, f"load_{k}")})
names = list(R)

# ------------------------------------------------ Fig: heterogeneity and gains
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.25]))
rng = np.random.default_rng(1)
for i, n in enumerate(names):
    pop = LOAD[n]()
    sh = pop.pi_org
    ax[0].scatter(i + rng.uniform(-0.28, 0.28, len(sh)), 100 * sh, s=6, color=COL[n], alpha=0.55, lw=0)
    q = np.percentile(100 * sh, [25, 50, 75])
    ax[0].plot([i - 0.33, i + 0.33], [q[1]] * 2, color="#0b0b0b", lw=1.6)
    ax[0].plot([i, i], [q[0], q[2]], color="#0b0b0b", lw=1.0)
ax[0].set_xticks(range(len(names)), [LABEL[n].replace(", Uganda", "").replace("Islands", "Is.") for n in names])
ax[0].set_ylabel("Household organic share (%)"); ax[0].set_ylim(-3, 103)
ax[0].set_title("(a) Between-household variation", loc="left", fontsize=9)
off = {"WEIGHT": -0.22, "SCORE": 0.0, "RESID": 0.22}
for i, n in enumerate(names):
    pb = R[n]["popboot"]["re_q"]
    for k in ("WEIGHT", "SCORE", "RESID"):
        q = np.array(pb[k])[:, 0]; est = R[n]["main"]["3,10"]["re"][k][0]
        ax[1].plot([i + off[k]] * 2, [q[0], q[2]], color=PCOL[k], lw=1.6, solid_capstyle="round")
        ax[1].scatter([i + off[k]], [est], s=26, color=PCOL[k], edgecolor="white", lw=1, zorder=3,
                      label=PLAB[k] if i == 0 else None)
ax[1].axhline(1, color="#52514e", lw=0.8, ls="--")
ax[1].set_xticks(range(len(names)), [LABEL[n].replace(", Uganda", "").replace("Islands", "Is.") for n in names])
ax[1].set_ylabel("Relative efficiency, organic share")
ax[1].set_title("(b) Gain from ranking, $m=3$, 30 sorted", loc="left", fontsize=9)
ax[1].legend(loc="upper right", fontsize=7.5, handletextpad=0.3)
fig.tight_layout(); fig.savefig("figures/fig_validation.pdf"); fig.savefig("figures/fig_validation.png", dpi=200)

# ------------------------------------------------ Fig: planning rule
fig, ax = plt.subplots(figsize=(3.6, 3.4))
sim = json.load(open("simulation/planning2.json" if os.path.exists("simulation/planning2.json") else "planning2.json"))["rows"]
sp = [v for r in sim for v in r["pred_n"]]; so = [v for r in sim for v in r["sim"]]
ax.scatter(sp, so, s=10, color="#b5b4ae", lw=0, label="Simulation")
for n in names:
    pr, ob = [], []
    for m, d in ((2, "2,15"), (3, "3,10"), (4, "4,8")):
        for k in R[n]["describe"]["proxies"]:
            pr += R[n]["pred"][str(m)][k]; ob += R[n]["main"][d]["re"][k]
    ax.scatter(pr, ob, s=16, marker=MK[n], color=COL[n], edgecolor="white", lw=0.5, label=LABEL[n].replace(", Uganda", ""))
lim = [0.9, max(2.1, max(so) + 0.05)]
ax.plot(lim, lim, color="#52514e", lw=0.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Predicted RE (planning rule)"); ax.set_ylabel("Observed RE")
ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.2)
fig.tight_layout(); fig.savefig("figures/fig_planning.pdf"); fig.savefig("figures/fig_planning.png", dpi=200)
print("figures written")

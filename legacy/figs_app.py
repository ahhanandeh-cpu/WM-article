import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "legend.frameon": False})
LAB = {"WEIGHT": "Net weight", "HHSIZE": "Household size", "SCORE": "Visual score",
       "ORGMASS": "Visual organic mass", "RESID": "Estimated food residual"}
S = json.load(open("app_results.json")); F = S["fractions"]; p = np.array(S["p"])
FN = ["Organics", "Paper", "Plastics", "Metals", "Glass", "Hygiene", "Other"]

# ---- Fig 5: sampling distributions of organic and metal shares, m=3, r=10 ----
E = np.load("app_est_m3.npy", allow_pickle=True).item()
col = {"SRS": "#8a8984", "WEIGHT": "#2a78d6", "RESID": "#eda100"}
lab = {"SRS": "Simple random sampling", "WEIGHT": "RSS, net weight", "RESID": "RSS, estimated food residual"}
ls = {"SRS": "--", "WEIGHT": "-", "RESID": "-"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, j, name in ((axes[0], 0, "Organics"), (axes[1], 3, "Metals")):
    lo = min(np.quantile(E[k][:, j], 0.001) for k in col); hi = max(np.quantile(E[k][:, j], 0.999) for k in col)
    x = np.linspace(lo, hi, 400)
    for k in col:
        ax.plot(x * 100, gaussian_kde(E[k][:, j])(x) / 100, color=col[k], ls=ls[k], lw=1.8, label=lab[k])
    ax.axvline(p[j] * 100, color="#0b0b0b", lw=0.9)
    ax.text(p[j] * 100, ax.get_ylim()[1] * 0.97, " true share", fontsize=7.5, va="top", color="#0b0b0b")
    ax.set_xlabel(f"Estimated {name.lower()} share (% of mass)"); ax.set_yticks([])
    ax.set_title(name, fontsize=9)
axes[0].set_ylabel("Density over 20,000 audits")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("fig5_app_dist.pdf"); fig.savefig("fig5_app_dist.png", dpi=200)

# ---- Fig 6: planning rule, predicted vs observed RE (simulation + application) ----
D = {2: 0.3183, 3: 0.4775, 4: 0.5739}
P2 = json.load(open("planning2.json"))
sx = [v for r in P2["rows"] for v in r["pred_n"]]; sy = [v for r in P2["rows"] for v in r["sim"]]
ax_, ay_ = [], []
for key, R in S["resampling"].items():
    m = R["m"]
    for k, rho in S["rho_pop"].items():
        ax_ += [1 / (1 - r * r * D[m]) for r in rho]; ay_ += R["re"][k]
fig, ax = plt.subplots(figsize=(4.0, 3.4))
ax.plot([0.9, 2.4], [0.9, 2.4], color="#8a8984", lw=0.9, ls="--")
ax.scatter(sx, sy, s=14, color="#2a78d6", edgecolor="white", lw=0.6, label="Simulation (192 cases)", zorder=3)
ax.scatter(ax_, ay_, s=16, marker="D", color="#eb6834", edgecolor="white", lw=0.6, label="Solomon Islands audit (105 cases)", zorder=3)
ax.set_xlim(0.9, 2.4); ax.set_ylim(0.9, 2.4)
ax.set_xlabel("Predicted RE, Eq. (6)"); ax.set_ylabel("Observed RE")
ax.legend(loc="lower right", fontsize=7.5)
fig.tight_layout(); fig.savefig("fig6_planning.pdf"); fig.savefig("fig6_planning.png", dpi=200)
print("app planning MAE", np.mean(np.abs(np.array(ax_) - ay_)), "corr", np.corrcoef(ax_, ay_)[0, 1],
      "sim MAE", np.mean(np.abs(np.array(sx) - sy)), "n", len(ax_), len(sx))

# ---- Tables ----
sites = ["Guadalcanal", "Arabella", "Kilusakwalo", "Ambu"]
L = [f"All households ({S['N']}) & {S['total_kg']:.0f} & " + " & ".join(f"{v*100:.1f}" for v in p) + " \\\\", "\\midrule"]
for s_ in sites:
    L.append(f"\\quad {s_} ({S['sites'][s_]}) & & " + " & ".join(f"{v*100:.1f}" for v in S["site_comp"][s_]) + " \\\\")
L.append("\\midrule")
L.append("Units with none (\\%) & & " + " & ".join(f"{v*100:.0f}" for v in S["zero_freq"]) + " \\\\")
L.append("Spearman $\\rho$(share, weight) & & " + " & ".join(f"{v:+.2f}" for v in S["share_weight_spearman"]) + " \\\\")
open("tab_app_comp.tex", "w").write("\n".join(L))

L = []
for key in ("2,15", "3,10", "4,8"):
    R = S["resampling"][key]
    L.append("\\midrule\n\\multicolumn{9}{l}{\\textit{$m=%d$, $r=%d$: %d households sorted, %d inspected}}\\\\" % (R["m"], R["r"], R["n"], R["inspected"]))
    for k in ("WEIGHT", "HHSIZE", "SCORE", "ORGMASS", "RESID"):
        L.append(f"{LAB[k]} & " + " & ".join(f"{v:.2f}" for v in R["re"][k]) + f" & {R['pwre'][k]:.2f} \\\\")
open("tab_app_re.tex", "w").write("\n".join(L))

# planning table (simulation) with normal-score rho
L = []
for g in (0.0, 1.0):
    L.append("\\midrule\n\\multicolumn{8}{l}{\\textit{Scenario %s}}\\\\" % ("A" if g == 0 else "B"))
    for o in P2["rows"]:
        if o["gamma"] == g and o["m"] == 4:
            L.append(LAB[o["design"]] + " & $\\hat\\rho$ & " + " & ".join(f"{r:+.2f}" for r in o["rho_n"]) + " \\\\")
            L.append(" & Predicted / simulated RE & " + " & ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(o["pred_n"], o["sim"])) + " \\\\")
open("tab_plan.tex", "w").write("\n".join(L))

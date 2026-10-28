import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 600})
COL = {"WEIGHT": "#2a78d6", "SCORE": "#eb6834", "ORGMASS": "#1baf7a", "RESID": "#eda100"}
MK = {"WEIGHT": "o", "SCORE": "s", "ORGMASS": "^", "RESID": "D"}
LAB = {"WEIGHT": "Net weight", "SCORE": "Visual score", "ORGMASS": "Visual organic mass",
       "RESID": "Estimated food residual"}
d = json.load(open("sim_results.json"))
cells = d["cells"]

def get(g, sv, design, m):
    for c in cells:
        if c["gamma"] == g and c["sv"] == sv and c["design"] == design and c["m"] == m:
            return c

# ---------- Figure 2: RE vs set size ----------
fr_idx = [(0, "Food/organic"), (1, "Paper/cardboard"), (2, "Plastics")]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey=True)
for row, g in enumerate((0.0, 1.0)):
    for col, (k, name) in enumerate(fr_idx):
        ax = axes[row, col]
        ax.axhline(1, color="#8a8984", lw=0.8, ls="--")
        for des in COL:
            re = [get(g, 0.10, des, m)["re"][k] for m in (2, 3, 4, 5)]
            ax.plot([2, 3, 4, 5], re, color=COL[des], marker=MK[des], ms=4, lw=1.6, label=LAB[des])
        if row == 0: ax.set_title(name, fontsize=9)
        if col == 0: ax.set_ylabel(("No share–weight link" if g == 0 else "Shares vary with weight") + "\nRelative efficiency")
        if row == 1: ax.set_xlabel("Set size $m$")
        ax.set_xticks([2, 3, 4, 5]); ax.set_ylim(0.9, 1.65)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("fig2_re_setsize.pdf"); fig.savefig("fig2_re_setsize.png", dpi=200)

# ---------- Figure 3: policy-weighted RE vs ranking noise ----------
def pwre(c_rss, c_srs, p):
    p = np.array(p)
    return (np.array(c_srs["mse"]) / p**2).sum() / (np.array(c_rss["mse"]) / p**2).sum()

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, g in zip(axes, (0.0, 1.0)):
    ax.axhline(1, color="#8a8984", lw=0.8, ls="--")
    for des in COL:
        vals = [pwre(get(g, sv, des, 4), get(g, sv, "SRS", 4), get(g, sv, des, 4)["p"]) for sv in (0.05, 0.10, 0.20)]
        ax.plot([0.05, 0.10, 0.20], vals, color=COL[des], marker=MK[des], ms=4, lw=1.6, label=LAB[des])
    ax.set_title("No share–weight link" if g == 0 else "Shares vary with weight", fontsize=9)
    ax.set_xlabel("Visual-score error SD $\\sigma_v$")
    ax.set_xticks([0.05, 0.10, 0.20])
axes[0].set_ylabel("Policy-weighted RE ($m=4$)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("fig3_pwre_noise.pdf"); fig.savefig("fig3_pwre_noise.png", dpi=200)

# ---------- Figure 4: cost break-even ----------
ratio = np.linspace(0, 0.15, 200)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
seq = ["#9ec5f0", "#5d9be3", "#2a78d6", "#1a4f93"]
for m, c in zip((2, 3, 4, 5), seq):
    ax.plot(ratio, 1 + m * ratio, color=c, lw=1.8)
    ax.text(0.151, 1 + m * 0.15, f"$m={m}$", color="#52514e", va="center", fontsize=8)
ax.axhspan(1.43, 1.63, color="#e6e5e0", alpha=0.6, lw=0)
ax.text(0.003, 1.53, "Best-proxy RE, $m=4$–5", fontsize=7.5, color="#52514e", va="center")
ax.set_xlim(0, 0.17); ax.set_ylim(1, 1.8)
ax.set_xlabel("Ranking cost / sorting cost per unit, $c_r/c_s$")
ax.set_ylabel("Break-even relative efficiency")
fig.tight_layout()
fig.savefig("fig4_breakeven.pdf"); fig.savefig("fig4_breakeven.png", dpi=200)

# ---------- tables (LaTeX snippets) ----------
FRN = ["Food", "Paper", "Plastic", "Glass", "Metal", "Other"]
lines = []
for g in (0.0, 1.0):
    lines.append("\\midrule\n\\multicolumn{9}{l}{\\textit{%s}}\\\\" % ("Scenario A: shares independent of load weight ($\\gamma=0$)" if g == 0 else "Scenario B: shares vary with load weight ($\\gamma=1$)"))
    for m in (2, 3, 4, 5):
        for j, des in enumerate(COL):
            c = get(g, 0.10, des, m)
            pw = pwre(c, get(g, 0.10, "SRS", m), c["p"])
            lead = f"{m}" if j == 0 else ""
            lines.append(f"{lead} & {LAB[des]} & " + " & ".join(f"{x:.2f}" for x in c["re"]) + f" & {pw:.2f}\\\\")
open("tab_re.tex", "w").write("\n".join(lines))

tc = json.load(open("tcov_results.json"))
rows = []
for g in (0.0, 1.0):
    for r in (5, 10, 20):
        for des in ("SRS", "WEIGHT", "RESID"):
            c = [x for x in tc if x["gamma"] == g and x["r"] == r and x["design"] == des][0]
            w = [x for x in cells if False]
            rows.append(f"{'A' if g==0 else 'B'} & {r} & {3*r} & {'SRS' if des=='SRS' else LAB[des]} & " + " & ".join(f"{x:.3f}" for x in c["cover"]) + "\\\\")
open("tab_cov.tex", "w").write("\n".join(rows))
p = d["ptrue"]
print({k: np.round(v, 3) for k, v in p.items()})
# max bias
print("max |relbias|", max(abs(x) for c in cells for x in c["relbias"]))

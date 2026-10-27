"""LaTeX tables (fragments in tables/) from results/<dataset>.json for every dataset present."""
import json, os
import numpy as np

ORDER = ["solomon", "vanuatu", "belize", "kampala"]
LABEL = {"solomon": "Solomon Islands", "vanuatu": "Vanuatu", "belize": "Belize", "kampala": "Kampala, Uganda"}
INFO = {  # region, collection period, year, source key
    "solomon": ("Pacific (Melanesia)", "2 days", "2018", "cefas2020"),
    "vanuatu": ("Pacific (Melanesia)", "1--2 days", "2018", "cefas2019vu"),
    "belize": ("Central America", "1 coll.", "2019", "cefas2020bz"),
    "kampala": ("East Africa", "7 days", "n.r.", "katukiza2026"),
}
PNAME = {"WEIGHT": "Net weight", "HHSIZE": "Household size", "DENSITY": "Bulk density", "SCORE": "Visual score",
         "ORGMASS": "Visual organic mass", "RESID": "Estimated residual"}
FR = ["Organics", "Paper", "Plastics", "Metals", "Glass", "Hygiene", "Other"]
os.makedirs("tables", exist_ok=True)
R = {n: json.load(open(f"results/{n}.json")) for n in ORDER if os.path.exists(f"results/{n}.json")}


def f2(x): return f"{x:.2f}"
def sgn(x): return f"{x:+.2f}".replace("-", "$-$")


def frac_cells(res, vec):
    F = res["describe"]["fractions"]
    return [f2(vec[F.index(f)]) if f in F else "--" for f in FR]


# ---------------------------------------------------------------- Table: datasets
L = []
for n, res in R.items():
    d = res["describe"]; F = d["fractions"]; reg, days, yr, key = INFO[n]
    org = 100 * d["p"][0]; met = 100 * d["p"][F.index("Metals")]; pla = 100 * d["p"][F.index("Plastics")]
    L.append(f"{LABEL[n]} & {yr} & {days} & {d['N']} & {len(d['strata'] or [])} & {d['total_kg']:,.0f} & "
             f"{d['unit_mass']['mean']:.1f} ({d['unit_mass']['cv']:.2f}) & {org:.1f} & {d['org_share_sd']:.2f} & "
             f"{pla:.1f} & {met:.1f} & {sgn(d['share_weight_spearman'][0])}\\\\")
tot = sum(r["describe"]["N"] for r in R.values())
open("tables/datasets.tex", "w").write("\n".join(L) + f"\n\\midrule\nTotal & & & {tot} & {sum(len(r['describe']['strata'] or []) for r in R.values())} & {sum(r['describe']['total_kg'] for r in R.values()):,.0f} & & & & & & \\\\\n")

# ---------------------------------------------------------------- Table: RE at m=3 (main)
def re_table(design, fname):
    L = []
    for n, res in R.items():
        o = res["main"][design]
        L.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{LABEL[n]}}} ($N={res['describe']['N']}$)}}\\\\")
        for k in res["describe"]["proxies"]:
            L.append(f"{PNAME[k]} & " + " & ".join(frac_cells(res, o["re"][k])) + f" & {f2(o['pwre'][k])}\\\\")
        if n != list(R)[-1]: L.append("\\midrule")
    open(f"tables/{fname}", "w").write("\n".join(L) + "\n")


re_table("3,10", "re_m3.tex"); re_table("2,15", "re_m2.tex"); re_table("4,8", "re_m4.tex")

# ---------------------------------------------------------------- Table: headline organics with uncertainty
L = []
for n, res in R.items():
    pb = res["popboot"]; F = res["describe"]["fractions"]
    row = [LABEL[n]]
    for k in ("SCORE", "RESID"):
        q = np.array(pb["re_q"][k])[:, 0]
        row.append(f"{res['main']['3,10']['re'][k][0]:.2f} [{q[0]:.2f}, {q[2]:.2f}]")
    row.append(f"{res['pred']['3']['RESID'][0]:.2f}")
    for d in ("2,15", "3,10", "4,8"):
        row.append(f"{100 * (1 - 1 / res['main'][d]['re']['RESID'][0]):.0f}")
    jm = F.index("Metals")
    row.append(f"{res['main']['3,10']['re']['RESID'][jm]:.2f}")
    qw = np.array(pb["re_q"]["WEIGHT"])[:, 0]
    row.insert(1, f"{res['main']['3,10']['re']['WEIGHT'][0]:.2f} [{qw[0]:.2f}, {qw[2]:.2f}]")
    L.append(" & ".join(row) + "\\\\")
open("tables/headline.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Supplement: coverage, bias, MC s.e.
L = []
for n, res in R.items():
    for d in ("2,15", "3,10", "4,8"):
        o = res["main"][d]
        for k in ("SRS", "WEIGHT", "RESID"):
            L.append(f"{LABEL[n] if (d=='2,15' and k=='SRS') else ''} & {d.split(',')[0]} & {PNAME.get(k, 'SRS')} & "
                     + " & ".join(f"{v:.2f}" if v == v else '--' for v in
                                  [o['cover'][k][res['describe']['fractions'].index(f)] if f in res['describe']['fractions'] else float('nan') for f in FR])
                     + f" & {100 * max(abs(x) for x in o['relbias'][k]):.1f}\\\\")
    if n != list(R)[-1]: L.append("\\midrule")
open("tables/coverage.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Supplement: sensitivity (m=3)
L = []
for n, res in R.items():
    c0 = res["c0"]
    first = True
    for s in res["sens"]:
        L.append(f"{LABEL[n] if first else ''} & 0--{s['levels']} & {s['sv']:.2f} & {s['c0']:.1f}{'$^\\ast$' if abs(s['c0']-c0)<1e-9 else ''} & "
                 f"{f2(s['re']['SCORE'][0])} & {f2(s['re']['ORGMASS'][0])} & {f2(s['re']['RESID'][0])} & {f2(s['pwre']['RESID'])}\\\\")
        first = False
    if n != list(R)[-1]: L.append("\\midrule")
open("tables/sensitivity.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Supplement: robustness variants
L = []
for n, res in R.items():
    F = res["describe"]["fractions"]; jm = F.index("Metals")
    m3 = res["main"]["3,10"]
    nr = res["noreplace"].get("3,10") or res["noreplace"].get("2,15")
    nrlab = "3" if "3,10" in res["noreplace"] else "2"
    st = res["stratified"]
    L.append(f"{LABEL[n]} & {f2(m3['re']['RESID'][0])} & {f2(m3['re']['RESID'][jm])} & "
             f"{f2(res['c0_05']['re']['RESID'][0])} & "
             f"{(f2(nr['re']['RESID'][0]) + ' ($m=' + nrlab + '$)') if nr else '--'} & "
             f"{f2(st['re']['RESID'][0])} & {f2(st['re']['RESID'][jm])} & {len(st['r_h'])}\\\\")
open("tables/robust.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Pilot selection
L = []
for n, res in R.items():
    for pol, lab in (("organics", "Organics only"), ("equal", "All fractions equal")):
        for rule in ("naive", "adjusted"):
            o = res["pilot"][f"{rule}:{pol}"]; fr = o["freq"]; tot = sum(fr.values())
            top = sorted(fr.items(), key=lambda t: -t[1])
            choice = "; ".join(f"{PNAME.get(k, 'SRS')} {100 * v / tot:.0f}\\%" for k, v in top[:3])
            L.append(f"{LABEL[n] if (pol=='organics' and rule=='naive') else ''} & {lab if rule=='naive' else ''} & "
                     f"{'Raw' if rule=='naive' else 'Adjusted'} & {choice} & {o['mean_realised']:.2f} ({o['best_attainable']:.2f}) & "
                     f"{o['mean_cost_ratio']:.2f} & {100 * o['p_cost_gt1']:.0f}\\\\")
    if n != list(R)[-1]: L.append("\\midrule")
open("tables/pilot.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- planning rule accuracy across datasets
pr, ob = [], []
for n, res in R.items():
    for m, d in ((2, "2,15"), (3, "3,10"), (4, "4,8")):
        for k in res["describe"]["proxies"]:
            pr += res["pred"][str(m)][k]; ob += res["main"][d]["re"][k]
pr, ob = np.array(pr), np.array(ob)
json.dump(dict(n=len(pr), corr=float(np.corrcoef(pr, ob)[0, 1]), mad=float(np.abs(pr - ob).mean()),
               maxad=float(np.abs(pr - ob).max())), open("tables/planning_accuracy.json", "w"), indent=1)
print(open("tables/planning_accuracy.json").read())

# ---------------------------------------------------------------- Supplement: full compositions
L = []
for n, res in R.items():
    d = res["describe"]; F = d["fractions"]
    cell = lambda vec, fmt: [fmt(vec[F.index(f)]) if f in F else "--" for f in FR]
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{LABEL[n]}}} ($N={d['N']}$, {d['total_kg']:,.0f} kg)}}\\\\")
    L.append("Mass share (\\%) & " + " & ".join(cell(d["p"], lambda x: f"{100*x:.1f}")) + "\\\\")
    L.append("Units with none (\\%) & " + " & ".join(cell(d["zero_freq"], lambda x: f"{100*x:.0f}")) + "\\\\")
    L.append("$\\rho_S$(share, $W$) & " + " & ".join(cell(d["share_weight_spearman"], sgn)) + "\\\\")
    if n != list(R)[-1]: L.append("\\midrule")
open("tables/composition.tex", "w").write("\n".join(L) + "\n")

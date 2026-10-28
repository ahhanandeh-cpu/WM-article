import json, numpy as np
P=json.load(open("planning.json")); D={int(k):v for k,v in P["D"].items()}
LAB={"WEIGHT":"Net weight","SCORE":"Visual score","ORGMASS":"Visual organic mass","RESID":"Estimated food residual"}
L=[]
for g in (0.0,1.0):
  L.append("\\midrule\n\\multicolumn{8}{l}{\\textit{Scenario %s}}\\\\"%("A" if g==0 else "B"))
  for o in P["rows"]:
    if o["gamma"]==g and o["m"]==4:
      L.append(LAB[o["design"]]+" & $\\hat\\rho$ & "+" & ".join(f"{r:+.2f}" for r in o["rho"][:6])+" & \\\\")
      L.append(" & Predicted / simulated RE & "+" & ".join(f"{a:.2f}/{b:.2f}" for a,b in zip(o["pred"],o["sim"]))+" & \\\\")
open("tab_plan.tex","w").write("\n".join(L))
G=[]
for rho in (0.3,0.4,0.5,0.6,0.7,0.8,0.9):
  G.append(f"{rho:.1f} & "+" & ".join(f"{1/(1-rho**2*D[m]):.2f}" for m in (2,3,4,5))+" \\\\")
open("tab_grid.tex","w").write("\n".join(G))
print(open("tab_grid.tex").read()); print(D)

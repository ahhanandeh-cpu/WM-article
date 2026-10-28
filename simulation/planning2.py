"""Compare planning-rule correlations: Spearman vs Pearson(d, normal score of proxy)."""
import numpy as np, json
from scipy import stats
from sim import draw, true_p, proxy
P=json.load(open("planning.json")); D={int(k):v for k,v in P["D"].items()}
sim=json.load(open("sim_results.json"))["cells"]
def nscore(x):
    r=stats.rankdata(x); return stats.norm.ppf(r/(len(x)+1))
rng=np.random.default_rng(3); rows=[]
for g in (0.0,1.0):
    p=true_p(g); W,pi,S=draw(rng,(400000,),g,0.10); d=W[:,None]*pi-p*W[:,None]
    for des in ("WEIGHT","SCORE","ORGMASS","RESID"):
        x=proxy(des,W,S)+1e-9*rng.random(W.shape); z=nscore(x)
        rs=[stats.spearmanr(x,d[:,k])[0] for k in range(6)]
        rn=[np.corrcoef(z,d[:,k])[0,1] for k in range(6)]
        for m in (2,3,4,5):
            s=[c for c in sim if c["gamma"]==g and c["sv"]==0.10 and c["design"]==des and c["m"]==m][0]["re"]
            rows.append(dict(gamma=g,design=des,m=m,rho_s=rs,rho_n=rn,
                pred_s=[1/(1-r*r*D[m]) for r in rs],pred_n=[1/(1-r*r*D[m]) for r in rn],sim=s))
ps=np.array([v for r in rows for v in r["pred_s"]]); pn=np.array([v for r in rows for v in r["pred_n"]]); sm=np.array([v for r in rows for v in r["sim"]])
print("Spearman  MAE",np.abs(ps-sm).mean().round(4),"corr",np.corrcoef(ps,sm)[0,1].round(3))
print("NormScore MAE",np.abs(pn-sm).mean().round(4),"corr",np.corrcoef(pn,sm)[0,1].round(3), "max abs",np.abs(pn-sm).max().round(3))
json.dump(dict(D=D,rows=rows),open("planning2.json","w"),indent=1)

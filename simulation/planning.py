import numpy as np, json
from scipy import stats, integrate
from sim import draw, true_p, proxy
def Dm(m):
    # (1/m) sum E[Z_(i:m)]^2 for standard normal order stats
    a=[]
    for i in range(1,m+1):
        f=lambda z: z*stats.norm.pdf(z)*m*stats.binom.pmf(i-1,m-1,stats.norm.cdf(z))
        a.append(integrate.quad(f,-10,10)[0])
    return np.mean(np.array(a)**2)
D={m:Dm(m) for m in (2,3,4,5)}; print({m:round(v,4) for m,v in D.items()})
sim=json.load(open("sim_results.json"))["cells"]
rng=np.random.default_rng(3); out=[]
for g in (0.0,1.0):
  p=true_p(g)
  W,pi,S=draw(rng,(1_000_000,),g,0.10)
  d=W[:,None]*pi - p*W[:,None]
  for des in ("WEIGHT","SCORE","ORGMASS","RESID"):
    x=proxy(des,W,S)
    # use Spearman (rank) correlation since only ranks matter
    rho=[stats.spearmanr(x[:200000],d[:200000,k])[0] for k in range(6)]
    for m in (2,3,4,5):
      pred=[1/(1-r**2*D[m]) for r in rho]
      simv=[c for c in sim if c["gamma"]==g and c["sv"]==0.10 and c["design"]==des and c["m"]==m][0]["re"]
      out.append(dict(gamma=g,design=des,m=m,rho=rho,pred=pred,sim=simv))
    print(g,des,np.round(rho,3))
json.dump(dict(D=D,rows=out),open("planning.json","w"),indent=1)
pr=np.array([r for o in out for r in o["pred"]]); sm=np.array([r for o in out for r in o["sim"]])
print("corr",np.corrcoef(pr,sm)[0,1],"mean abs diff",np.mean(np.abs(pr-sm)),"mean ratio sim/pred",np.mean((sm-1+1e-9)/(pr-1+1e-9)))
for o in out:
  if o["m"]==4: print(o["gamma"],o["design"],np.round(o["pred"],2),np.round(o["sim"],2))

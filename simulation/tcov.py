import numpy as np, json
from scipy.stats import t
from sim import one_cell, true_p
rng=np.random.default_rng(11); out=[]
for g in (0.0,1.0):
  p=true_p(g)
  for r in (5,10,20):
    o=one_cell(3,r,g,0.10,10000,rng,p)
    for d,(e,v) in o.items():
      df = 3*r-1 if d=="SRS" else 3*(r-1)
      c=(np.abs(e-p)<=t.ppf(.975,df)*np.sqrt(v)).mean(0)
      out.append(dict(gamma=g,r=r,design=d,df=df,cover=c.tolist())); print(g,r,d,np.round(c,3))
json.dump(out,open("tcov_results.json","w"),indent=1)

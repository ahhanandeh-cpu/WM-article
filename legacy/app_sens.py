import numpy as np, json, app
from app import one_audit, estimate, p
out={}
for sv,c0 in ((0.05,0.5),(0.10,0.5),(0.20,0.5),(0.30,0.5),(0.10,0.7),(0.10,0.3)):
    app.C0=c0
    rng=np.random.default_rng(int(sv*100+c0*10))
    est={k:[] for k in ["SRS","SCORE","ORGMASS","RESID"]}
    for _ in range(10000):
        a=one_audit(rng,3,10,sv)
        for k in est: est[k].append(estimate(a[k][0]))
    mse={k:((np.array(v)-p)**2).mean(0) for k,v in est.items()}
    out[f"{sv},{c0}"]={k:(mse["SRS"]/mse[k]).tolist() for k in est}
    print(sv,c0,{k:np.round(mse["SRS"][:4]/mse[k][:4],2).tolist() for k in est if k!="SRS"},flush=True)
json.dump(out,open("app_sens.json","w"),indent=1)

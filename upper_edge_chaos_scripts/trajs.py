from paths import data
import numpy as np, sys, pickle
from probe import mirror, probe
from physics.solve import run_trajectory
E=0.03350081271247681
def dense(b,m,rt=1e-12):
    r=run_trajectory(E,b,m[0],m[1],start_radius=64000.,stop='escape',rtol=rt,atol=rt*1e-2,method='reference',keep_solution=True)
    sol=r['solution']; t=sol.t
    rr=np.linalg.norm(sol.y[0:3],axis=0); k=np.where(rr<12)[0]
    t0,t1=t[k[0]],min(t[-1],t[k[-1]]+60)
    tt=np.linspace(t0,t1,int((t1-t0)*20)+2); Y=sol.sol(tt)
    return dict(b=b,outcome=r['outcome'],t=tt-tt[0],y=Y[:15])
res={}
mB=mirror(np.pi/4,-0.07*np.pi); be=float(np.load(data('stairB_1e-13.npz'))['be'])
res['B']=[dense(be-dx,mB,1e-13) for dx in [1e-2,1e-3,5e-5,2e-6,8e-8,3e-9]]+[dense(be+1e-4,mB,1e-13)]
for d in res['B']: print('B',d['b']-be,d['outcome'],round(d['t'][-1]))
d=np.load(data('cA2_12.npz')); o,n,b=d['out'],d['npk'],d['b']
cand=[i for i in range(len(b)-1) if o[i]!=o[i+1] and min(n[i],n[i+1])>=12]
i=cand[len(cand)//2]; print('A pair',b[i],b[i+1],o[i],o[i+1],n[i],n[i+1])
mA=mirror(np.pi/4,-0.042*np.pi)
res['A']=[dense(b[i],mA),dense(b[i+1],mA)]
for x in res['A']: print('A',repr(x['b']),x['outcome'],round(x['t'][-1]))
pickle.dump(res,open(data('trajs.pkl'),'wb'))

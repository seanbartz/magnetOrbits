import numpy as np, pickle
from physics.solve import run_trajectory
E=0.03350081271247681; I=0.8
import sys; sys.path.insert(0,'scratch'); from probe import mirror
m=mirror(np.pi/4,-0.042*np.pi)
T=pickle.load(open('scratch/trajs.pkl','rb'))
for tr in T['A']:
    r=run_trajectory(E,tr['b'],*m,start_radius=64000.,stop='escape',rtol=1e-12,atol=1e-14,method='reference',keep_solution=True)
    sol=r['solution']; y=sol.y; rr=np.linalg.norm(y[0:3],axis=0)
    Kr=0.5*I*((y[12:15]**2).sum(0)+(y[15:18]**2).sum(0)); Kt=0.5*(y[3:6]**2).sum(0)
    rr_=rr.copy(); 
    sel=np.where((rr>6)&(sol.t>sol.t[np.argmax(rr<12)]))[0]
    if len(sel):
        k=sel[np.argmax(rr[sel])]
        print(r['outcome'],'apoapsis r=%.2f  Krot/E=%.3f  Ktrans/E=%.3f'%(rr[k],Kr[k]/E,Kt[k]/E))
    print('  max Krot/E during interaction',Kr[rr<12].max()/E)
    tr['y']=y[:, :]; 

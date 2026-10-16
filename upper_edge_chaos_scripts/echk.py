from paths import data
import numpy as np, pickle
from physics.solve import run_trajectory
E=0.03350081271247681
from probe import mirror, I
m=mirror(np.pi/4,-0.042*np.pi)
T=pickle.load(open(data('trajs.pkl'),'rb'))
for tr in T['A']:
    r=run_trajectory(E,tr['b'],*m,start_radius=64000.,stop='escape',rtol=1e-12,atol=1e-14,method='reference',keep_solution=True)
    sol=r['solution']; y=sol.y; rr=np.linalg.norm(y[0:3],axis=0)
    Kr=0.5*I*((y[12:15]**2).sum(0)+(y[15:18]**2).sum(0)); Kt=0.5*(y[3:6]**2).sum(0)
    inn=rr<12
    if not inn.any(): print(r['outcome'],'never inside 12R, skipped'); continue
    loc=np.where((rr[1:-1]>rr[:-2])&(rr[1:-1]>rr[2:]))[0]+1   # turning points, not the stop point
    sel=loc[(rr[loc]>6)&(loc>np.argmax(inn))]
    if len(sel):
        k=sel[np.argmax(rr[sel])]
        print(r['outcome'],'apoapsis r=%.2f  Krot/E=%.3f  Ktrans/E=%.3f'%(rr[k],Kr[k]/E,Kt[k]/E))
    print('  max Krot/E during interaction',Kr[inn].max()/E)

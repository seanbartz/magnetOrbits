from paths import data
import numpy as np, sys
from functools import partial
from multiprocessing import Pool
from probe import probe, mirror
mB=mirror(np.pi/4,-0.07*np.pi)
RT=float(sys.argv[1]); AT=RT*1e-2
def edge():
    lo,hi=3.28198,3.28199   # escape at lo, contact at hi
    if not (probe(lo,*mB,rtol=RT,atol=AT)[0]==0 and probe(hi,*mB,rtol=RT,atol=AT)[0]==1): sys.exit('edge outside bracket at this rtol')
    for _ in range(45):
        mid=(lo+hi)/2
        if probe(mid,*mB,rtol=RT,atol=AT)[0]==0: lo=mid
        else: hi=mid
    return lo
x=np.logspace(-11,-1.5,400)
def f(be,dx): return probe(be-dx,*mB,rtol=RT,atol=AT)[:2]
if __name__=='__main__':
    be=edge(); print('rtol',RT,'edge',repr(be))   # in __main__ only: spawned workers re-import this module
    with Pool(2) as p: R=np.array(p.map(partial(f,be),x))
    np.savez(data(f'stairB_{RT:.0e}.npz'),dx=x,out=R[:,0],npk=R[:,1],be=be)
    for k in range(1,20):
        sel=(R[:,0]==0)&(R[:,1]==k)
        if sel.any(): print(k,'%.3e .. %.3e'%(x[sel].min(),x[sel].max()))
    print('contacts',(R[:,0]==1).sum())

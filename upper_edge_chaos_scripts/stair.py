from paths import data
import numpy as np, sys
from functools import partial
from multiprocessing import Pool
from probe import probe, mirror
s=np.pi/4; dpi=-0.07
m1,m2=mirror(s,dpi*np.pi)
x=np.logspace(-12.5,-1.5,600)
def edge():
    lo,hi=3.281,3.283   # escape at lo, contact at hi
    assert probe(lo,m1,m2)[0]==0 and probe(hi,m1,m2)[0]==1
    for _ in range(60):
        mid=0.5*(lo+hi)
        if probe(mid,m1,m2)[0]==0: lo=mid
        else: hi=mid
        if hi-lo<1e-13: break
    print('edge',repr(lo), hi-lo); return lo
def f(be,dx): return probe(be-dx,m1,m2)[:2]
if __name__=='__main__':
    be=edge()   # in __main__ only: spawned workers re-import this module
    with Pool(2) as p: R=np.array(p.map(partial(f,be),x))
    np.savez(data('stairB.npz'),dx=x,out=R[:,0],npk=R[:,1],be=be)
    for k in range(1,40):
        sel=(R[:,0]==0)&(R[:,1]==k)
        if sel.any(): print(k, '%.3e .. %.3e'%(x[sel].min(),x[sel].max()))
    print('contacts below edge:',(R[:,0]==1).sum())

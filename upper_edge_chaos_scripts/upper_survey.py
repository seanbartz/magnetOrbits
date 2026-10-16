from paths import data
import numpy as np, sys
from multiprocessing import Pool
from probe import probe, mirror
s=np.pi/4
f=np.load(data('bd_s025.npz')); out=f['out']; ds=f['ds']; bs=f['bs']
def work(i):
    d=ds[i]; o=out[i]
    # window = escape run(s) below outer escape region; outer region = last run of escapes reaching b=6
    trans=np.where(np.diff(o)!=0)[0]
    if len(trans)<3: return None
    bu=bs[trans[1]]  # upper edge of first window (escape->contact)
    m1,m2=mirror(s,d)
    bb=np.linspace(bu-0.03,bu+0.04,1401)
    R=np.array([probe(b,m1,m2)[:2] for b in bb])
    nt=(np.diff(R[:,0])!=0).sum()
    return d/np.pi, bs[trans[0]], bu, nt, R[:,1].max(), (R[:,0]==2).sum()
if __name__=='__main__':
    idx=list(range(20,101,4))
    with Pool(2) as p: res=p.map(work,idx)
    for r in res:
        if r: print('d/pi=%.4f  window ~[%.3f, %.3f]  transitions near upper edge=%d  max periapses=%d timeouts=%d'%r)

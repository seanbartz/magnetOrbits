from paths import data
import numpy as np, sys, time
from multiprocessing import Pool
from probe import probe, mirror
s=np.pi/4
ds=np.linspace(-0.09*np.pi,0.01*np.pi,101)
bs=np.linspace(0.5,6.0,551)
def row(d):
    m1,m2=mirror(s,d); return [probe(b,m1,m2)[:2] for b in bs]
if __name__=='__main__':
    t=time.time()
    with Pool(2) as p: R=p.map(row,ds)
    R=np.array(R); np.savez(data('bd_s025.npz'),out=R[...,0],npk=R[...,1],ds=ds,bs=bs,s=s)
    print('s',time.time()-t, 'timeouts',(R[...,0]==2).sum())

from paths import data
import numpy as np, sys, time
from multiprocessing import Pool
from probe import probe, mirror
s=np.pi/4
d0,d1,b0,b1,nd,nb,name=[float(x) for x in sys.argv[1:7]]+[sys.argv[7]]
ds=np.linspace(d0*np.pi,d1*np.pi,int(nd)); bs=np.linspace(b0,b1,int(nb))
def row(d):
    m1,m2=mirror(s,d); return [probe(b,m1,m2)[:2] for b in bs]
if __name__=='__main__':
    t=time.time()
    with Pool(2) as p: R=np.array(p.map(row,ds))
    np.savez(data(f'{name}.npz'),out=R[...,0],npk=R[...,1],ds=ds,bs=bs,s=s)
    print('s',time.time()-t,'timeouts',(R[...,0]==2).sum())

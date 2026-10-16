from paths import data
import numpy as np, sys
from multiprocessing import Pool
from probe import probe, mirror
s=np.pi/4
dpi,b0,b1,n,name=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
m1,m2=mirror(s,dpi*np.pi)
bb=np.linspace(b0,b1,n)
import os
RT=float(os.environ.get("RT","1e-10"))
def f(b): return probe(b,m1,m2,rtol=RT,atol=RT*1e-2)[:3]
if __name__=='__main__':
    with Pool(2) as p: R=np.array(p.map(f,bb,chunksize=200))
    np.savez(data(f'{name}.npz'),b=bb,out=R[:,0],npk=R[:,1],t=R[:,2],dpi=dpi)
    o=R[:,0]; tr=np.where(np.diff(o)!=0)[0]
    print(name,'transitions',len(tr),'max npk',R[:,1].max(),'timeouts',(o==2).sum())
    print('edges:',np.round(bb[tr][:40],5))

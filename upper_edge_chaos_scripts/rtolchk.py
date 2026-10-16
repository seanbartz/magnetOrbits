from paths import data
import numpy as np, sys
from multiprocessing import Pool
from probe import probe, mirror
m1,m2=mirror(np.pi/4,-0.042*np.pi)
def f(b): return probe(b,m1,m2,rtol=1e-12,atol=1e-14)[:2]
if __name__=='__main__':
    for name in ['cA3','cA2']:
        d=np.load(data(f'{name}.npz')); idx=np.arange(0,len(d['b']),8)
        with Pool(2) as p: R=np.array(p.map(f,d['b'][idx],chunksize=50))
        print(name,'outcome flips',(R[:,0]!=d['out'][idx]).mean(),'npk changes',(R[:,1]!=d['npk'][idx]).mean())
    be=float(np.load(data('stairB.npz'))['be']); mB=mirror(np.pi/4,-0.07*np.pi)
    lo,hi=be-1e-8,be+1e-8
    for _ in range(40):
        mid=(lo+hi)/2
        if probe(mid,*mB,rtol=1e-12,atol=1e-14)[0]==0: lo=mid
        else: hi=mid
    print('edge B rtol1e-10',be,' rtol1e-12',lo, 'diff',lo-be)

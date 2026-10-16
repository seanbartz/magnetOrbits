import numpy as np, sys
sys.path.insert(0,'scratch')
from probe import probe, mirror, I
d=np.load('scratch/cA1_12.npz'); k=np.argsort(-d['npk'])[:3]
m=mirror(np.pi/4,-0.042*np.pi)
for b in d['b'][k]:
    out,npk,tc,tp,yp,y0=probe(b,*m,rtol=1e-12,atol=1e-14,keep=True)
    r=np.linalg.norm(yp[:,0:3],axis=1); th=np.arctan2(yp[:,1],yp[:,0])
    a1=np.degrees((np.arctan2(yp[:,7],yp[:,6])-th+np.pi)%(2*np.pi)-np.pi)
    print('b',repr(b),'out',out,'npk',npk)
    print('  r_peri',np.round(r,2).tolist()[:60])
    print('  dt',np.round(np.diff(tp),1).tolist()[:40])
    print('  a1',np.round(a1).tolist()[:40])

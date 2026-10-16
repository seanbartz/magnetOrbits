import numpy as np, sys
sys.path.insert(0,'scratch')
from probe import probe, mirror, I
s=np.pi/4; m1,m2=mirror(s,-0.07*np.pi); be=float(np.load('scratch/stairB.npz')['be'])
for dx in [1e-11, 1e-11*1.2]:
    out,npk,tc,tp,yp,y0=probe(be-dx,m1,m2,keep=True)
    print('dx',dx,'out',out,'npk',npk)
    for k in range(len(tp)):
        y=yp[k]; r=y[0:3]; v=y[3:6]; M1=y[6:9]; M2=y[9:12]; w1=y[12:15]; w2=y[15:18]
        rr=np.linalg.norm(r); th=np.arctan2(r[1],r[0])
        a1=np.degrees((np.arctan2(M1[1],M1[0])-th+np.pi)%(2*np.pi)-np.pi); a2=np.degrees((np.arctan2(M2[1],M2[0])-th+np.pi)%(2*np.pi)-np.pi)
        Lorb=r[0]*v[1]-r[1]*v[0]; Krot=0.5*I*(w1@w1+w2@w2)
        print(f"  peri {k:2d} t={tp[k]-tp[0]:8.1f} r={rr:.4f} a1={a1:7.1f} a2={a2:7.1f} w1z={w1[2]:+.4f} w2z={w2[2]:+.4f} Lorb={Lorb:.4f} Krot/E={Krot/0.0335008:.3f}")

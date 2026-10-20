import paths
import numpy as np, time
from physics.solve import run_trajectory
E=0.03350081271247681
def m(deg): a=np.radians(deg); return np.array([np.cos(a),np.sin(a),0.])
S=dict(start_radius=64000.,method='numba',stop='escape')
p1,p2=5.,155.
t=time.time()
bs=np.arange(2.5,4.6,0.005); o=[run_trajectory(E,b,m(p1),m(p2),**S)['outcome'][0] for b in bs]
print('s',time.time()-t)
s=''.join(o); ch=[round(bs[i]+0.0025,4) for i in range(len(s)-1) if s[i]!=s[i+1]]
print(ch)

import paths
import numpy as np
from physics.solve import run_trajectory
E=0.03350081271247681
def m(deg): a=np.radians(deg); return np.array([np.cos(a),np.sin(a),0.])
S=dict(start_radius=600/6.94,method='numba',stop='escape',asymptotic_correction=False)
bs=np.arange(2.5,4.6,0.01)
for p1,p2 in [(5,155),(185,335)]:
    s=''.join(run_trajectory(E,b,m(p1),m(p2),**S)['outcome'][0] for b in bs)
    print((p1,p2), s[0], [round(bs[i]+0.005,3) for i in range(len(s)-1) if s[i]!=s[i+1]])

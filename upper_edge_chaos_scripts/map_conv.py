import numpy as np
from physics.solve import run_trajectory
E=0.03350081271247681
def m(deg): a=np.radians(deg); return np.array([np.cos(a),np.sin(a),0.])
S=dict(start_radius=64000.,method='numba',stop='escape')
bs=np.arange(2.5,4.6,0.01)
for p1,p2 in [(5,155),(175,25),(185,335),(355,205),(5,205),(175,325),(25,175),(155,5)]:
    s=''.join(run_trajectory(E,b,m(p1),m(p2),**S)['outcome'][0] for b in bs)
    ch=[round(bs[i]+0.005,3) for i in range(len(s)-1) if s[i]!=s[i+1]]
    print((p1,p2), s[0], ch)

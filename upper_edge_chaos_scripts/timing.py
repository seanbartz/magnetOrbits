import numpy as np, time
from physics.solve import run_trajectory
E=0.03350081271247681
m1=np.array([np.cos(np.pi/4),np.sin(np.pi/4),0]); d=-0.03*np.pi; a2=np.pi-np.pi/4+d; m2=np.array([np.cos(a2),np.sin(a2),0])
run_trajectory(E,3.0,m1,m2,start_radius=64000.,method='numba',stop='escape')
for r0 in [64000.,1000.]:
    t=time.time()
    for b in np.linspace(1,5,100): run_trajectory(E,b,m1,m2,start_radius=r0,method='numba',stop='escape')
    print(r0,'ms/run',(time.time()-t)*10)

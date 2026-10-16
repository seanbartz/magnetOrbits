"""Outcome + periapsis count for one trajectory via the repo's compiled integrator
(same in-state and events as solve.run_trajectory(method='numba', stop='escape'))."""
import numpy as np
from physics import fast
from physics.solve import initial_state
from physics.model import SphereParams, CONTACT_SEPARATION
E0=0.03350081271247681; I=SphereParams().inertia_tilde
def mvec(phi): return np.array([np.cos(phi),np.sin(phi),0.0])
def mirror(s,d): return mvec(s), mvec(np.pi-s+d)
def probe(b, m1, m2, E=E0, r0=64000., rtol=1e-10, atol=1e-12, keep=False):
    y0=initial_state(E,b,m1,m2,start_radius=r0,inertia_tilde=I,quadrature='gauss')
    tmax=2*(2*r0)/np.sqrt(2*E)+2000.
    st,nrhs,tf,yf,cnt,evt,evy,stt,sd=fast.integrate_scattering(y0,float(tmax),I,rtol,atol,float(r0),True,False)
    rp=np.linalg.norm(evy[1,:cnt[1],0:3],axis=1) if cnt[1] else np.array([])
    graze=np.nonzero(rp<CONTACT_SEPARATION)[0]
    if cnt[0]>0 or len(graze):
        tc=min(list(evt[0,:cnt[0]])+[evt[1,k] for k in graze]); out=1
        npk=int(np.sum(evt[1,:cnt[1]]<tc))
    elif cnt[2]>0: out=0; npk=int(cnt[1]); tc=evt[2,0]
    else: out=2; npk=int(cnt[1]); tc=tf
    if keep: return out,npk,tc,evt[1,:cnt[1]].copy(),evy[1,:cnt[1]].copy(),y0
    return out,npk,tc

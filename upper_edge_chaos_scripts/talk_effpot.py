"""Slide figures: effective potentials (locked head-to-tail with co-rotating spin, and
moments perpendicular to the collision plane), and trajectories at one (E, b) for
different moment orientations. Writes to output/ (or UEC_OUT)."""
from paths import out
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from physics.solve import run_trajectory
from physics.model import SphereParams

plt.rcParams.update({'font.size':14,'axes.labelsize':14,'legend.fontsize':12,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#666',
                     'pdf.fonttype':42,'savefig.bbox':'tight'})
E=0.03350081271247681; B=5.0; I=SphereParams().inertia_tilde   # I = 4/5
J=B*np.sqrt(2*E)                                                 # non-spinning in-state: J = L_orb
COL={'locked':'#c0392b','anti':'#2a78d6','par':'#7a7a7a'}

def ueff(r,case,J=J):
    if case=='locked': return J**2/(2*(r**2+2*I))-2/r**3      # head-to-tail along r_hat, spins co-rotate
    if case=='anti':   return J**2/(2*r**2)-1/r**3            # +z/-z: attractive
    if case=='par':    return J**2/(2*r**2)+1/r**3            # +z/+z: repulsive

# ---------------- effective potential ----------------
r=np.linspace(2,14,2000)
fig,ax=plt.subplots(figsize=(8.5,5.4))
ax.axvspan(1.6,2,color='#000',alpha=0.08,lw=0)
lab={'locked':'locked head-to-tail (with spin)','anti':'out of plane, antiparallel','par':'out of plane, parallel'}
for c in ['locked','anti','par']:
    u=ueff(r,c)/E; ax.plot(r,u,color=COL[c],lw=2.2,label=lab[c])
    k=np.argmax(u);
    if 0<k<len(r)-1: ax.plot(r[k],u[k],'o',color=COL[c],ms=6)
ax.axhline(1,color='#222',lw=1.2,ls='--'); ax.text(13.9,1.06,'E',ha='right',fontsize=14)
ax.set_xlim(1.6,14); ax.set_ylim(-1.5,2.5)
ax.set_xlabel('r / R'); ax.set_ylabel('U_eff / E'); ax.text(1.8,-1.35,'contact',rotation=90,fontsize=11,color='#555',ha='center')
ax.legend(frameon=False,loc='lower right')
fig.savefig(out('effective_potential.pdf')); plt.close(fig)

# ---------------- trajectories at the same (E, b) ----------------
cases=[('locked','head-to-tail',(1,0,0),(1,0,0)),
       ('anti','out of plane, antiparallel',(0,0,1),(0,0,-1)),
       ('par','out of plane, parallel',(0,0,1),(0,0,1))]
fig,ax=plt.subplots(figsize=(8.5,6))
ax.add_patch(plt.Circle((0,0),2,fc='#eee',ec='#999',zorder=1))
for c,name,m1,m2 in cases:
    res=run_trajectory(E,B,np.array(m1,float),np.array(m2,float),start_radius=64000.,stop='r_out',outgoing_radius=40.,
                       rtol=1e-11,atol=1e-13,method='reference',keep_solution=True)
    sol=res['solution']; rr=np.linalg.norm(sol.y[0:3],axis=0)
    k=np.where(rr<25)[0]; t=np.linspace(sol.t[k[0]],sol.t[-1] if res['outcome']=='contact' else sol.t[k[-1]],4000)
    y=sol.sol(t); keep=np.linalg.norm(y[0:3],axis=0)<25
    ax.plot(y[0,keep],y[1,keep],color=COL[c],lw=2.2,label=f"{name}: {res['outcome']}",zorder=3)
    if res['outcome']=='contact': ax.plot(y[0,-1],y[1,-1],'X',color=COL[c],ms=11,zorder=4)
    print(c,res['outcome'])
ax.set_aspect('equal'); ax.set_xlim(-16,22); ax.set_ylim(-8,12)
ax.set_xlabel('x / R'); ax.set_ylabel('y / R'); ax.legend(frameon=False,loc='lower left',fontsize=12)
fig.savefig(out('orientation_trajectories.pdf')); plt.close(fig)

# barrier heights for the record
for c in ['locked','anti','par']:
    u=ueff(r,c); k=np.argmax(u); print(c,'max U_eff/E = %.3f at r = %.2f'%(u[k]/E,r[k]))

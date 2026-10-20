"""Instantaneous ("frozen") effective potential for free-rotor orbits.

At each time, freeze the orbital angular momentum L(t) and the moment angles relative to
r_hat. In the plane the dipole energy is U = -C(t)/r^3 with C = 3(m1.r)(m2.r) - m1.m2, so
    U_eff(r'; t) = L(t)^2 / 2r'^2 - C(t) / r'^3.
Exact identities (reduced mass 1):  1/2 rdot^2 + U_eff(r; t) = E - K_rot  and
rddot = -dU_eff/dr at fixed L, C. The system is drawn as a point at (r, E - K_rot).
Writes slide figures to output/ (or UEC_OUT)."""
from paths import data, out
import numpy as np, os, pickle, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from probe import mirror, I, E0
from physics.solve import run_trajectory

import talk_style
from matplotlib.colors import LinearSegmentedColormap
CMAP=LinearSegmentedColormap.from_list('t',['#1f3b73','#2a78d6','#3aa58a','#e0a020','#d9531e'])

def dense(b,m,rt=1e-12,rcut=14,tail=40,npt=25):
    cache=data(f'dense_{b:.12f}_{m[1][0]:.6f}_{rt:.0e}_{rcut}_{tail}_{npt}.pkl')
    if os.path.exists(cache): return pickle.load(open(cache,'rb'))
    r=run_trajectory(E0,b,m[0],m[1],start_radius=64000.,stop='escape',rtol=rt,atol=rt*1e-2,
                     method='reference',keep_solution=True)
    sol=r['solution']; rr=np.linalg.norm(sol.y[0:3],axis=0); k=np.where(rr<rcut)[0]
    t0,t1=sol.t[k[0]],min(sol.t[-1],sol.t[k[-1]]+tail)
    tt=np.linspace(t0,t1,int((t1-t0)*npt)+2)
    res=dict(b=b,outcome=r['outcome'],t=tt-tt[0],y=sol.sol(tt))
    pickle.dump(res,open(cache,'wb')); return res

def frozen(tr):
    y=tr['y']; r=y[0:3]; v=y[3:6]; rr=np.linalg.norm(r,axis=0); rh=r/rr
    L=r[0]*v[1]-r[1]*v[0]
    a1=(y[6:9]*rh).sum(0); a2=(y[9:12]*rh).sum(0); m12=(y[6:9]*y[9:12]).sum(0)
    C=3*a1*a2-m12
    Kr=0.5*I*((y[12:15]**2).sum(0)+(y[15:18]**2).sum(0))
    Erad=(E0-Kr)/E0
    rdot=(r*v).sum(0)/rr
    chk=np.max(abs(0.5*rdot**2+L**2/(2*rr**2)-C/rr**3-(E0-Kr)))/E0   # identity check
    return dict(t=tr['t'],r=rr,L=L,C=C,Erad=Erad,chk=chk)

def curve(F,k,rg): return (F['L'][k]**2/(2*rg**2)-F['C'][k]/rg**3)/E0

def draw(axL,axR,F,ks,xlim=(2,12),ylim=(-3,4),cmin=0,cmax=None):
    rg=np.linspace(2,xlim[1],600); t=F['t']; cmax=cmax or t[-1]
    col=lambda k:CMAP(np.clip((t[k]-cmin)/(cmax-cmin),0,1))
    for a in (axL,axR):
        a.axvspan(xlim[0]-0.4,2,color='#000',alpha=0.08,lw=0); a.axhline(0,color='#bbb',lw=0.8)
        a.set_xlim(xlim[0]-0.4,xlim[1]); a.set_ylim(*ylim); a.set_xlabel(r'$r/R$')
    for k in ks:
        axL.plot(rg,curve(F,k,rg),color=col(k),lw=2)
        axL.plot(F['r'][k],F['Erad'][k],'o',ms=9,color=col(k),mec='white',mew=1.2,zorder=5)
    axR.plot(F['r'],F['Erad'],color='#ccc',lw=0.8,zorder=1)
    w=(t>=cmin)&(t<=cmax)
    pts=axR.scatter(F['r'][w],F['Erad'][w],c=t[w],cmap=CMAP,s=3,lw=0,vmin=cmin,vmax=cmax,rasterized=True,zorder=2)
    cb=plt.colorbar(pts,ax=axR,pad=0.02); cb.set_label(r'time ($t_0$)')
    for k in ks: axR.plot(F['r'][k],F['Erad'][k],'o',ms=9,color=col(k),mec='white',mew=1.2,zorder=5)
    axL.set_ylabel(r'$U_{\rm eff}/E$'); axR.set_ylabel(r'$(E-K_{\rm rot})/E$')

def save(fig,name): fig.savefig(out(name)); plt.close(fig); print('wrote',name)

def tidx(F,cond): return int(np.where(cond)[0][0])

mB=mirror(np.pi/4,-0.07*np.pi); mA=mirror(np.pi/4,-0.042*np.pi)
be=float(np.load(data('stairB_1e-13.npz'))['be'])

if __name__=='__main__':
    # ---- 1. head-to-tail, b = 5R: capture (compare the static locked curve on slide 4) ----
    F=frozen(dense(5.0,(np.array([1.,0,0]),np.array([1.,0,0])),rcut=12,tail=30,npt=40)); print('h2t chk',F['chk'])
    r,t=F['r'],F['t']
    ks=[tidx(F,r<11),tidx(F,r<7),tidx(F,r<4.5),tidx(F,r<3),len(t)-1]
    fig,ax=plt.subplots(1,2,figsize=(14,5)); draw(ax[0],ax[1],F,ks,xlim=(2,12),ylim=(-2,3))
    save(fig,'frozen_ueff_head_to_tail.pdf')

    # ---- 2. escape window, d = -0.07pi: below (3.0R), inside (3.2R), above (3.4R) ----
    fig,ax=plt.subplots(2,3,figsize=(21,11.5))
    for j,(b0,lab) in enumerate([(3.0,'below'),(3.2,'inside'),(3.4,'above')]):
        F=frozen(dense(b0,mB,rcut=12,tail=30,npt=20)); r,t=F['r'],F['t']; print(b0,'chk',F['chk'])
        esc=r[-1]>3; kmin=int(np.argmin(r)) if esc else len(t)-1
        ks=[tidx(F,r<9),tidx(F,r<5),kmin]+([len(t)-1] if esc else [])
        draw(ax[0,j],ax[1,j],F,ks,xlim=(2,10),ylim=(-3,4))
        ax[0,j].set_title(rf'{lab}: $b={b0}R$, '+('escape' if esc else 'contact'))
        if j: ax[0,j].set_ylabel(''); ax[1,j].set_ylabel('')
    save(fig,'frozen_ueff_window.pdf')

    # ---- 3. rocking orbit, 3e-9 below the cut-B edge ----
    F=frozen(dense(be-3e-9,mB,rt=1e-13)); r,t=F['r'],F['t']; print('rock chk',F['chk'])
    ph=(t>50)&(t<190)
    peri=np.where((r[1:-1]<r[:-2])&(r[1:-1]<r[2:])&ph[1:-1])[0]+1
    apo=np.where((r[1:-1]>r[:-2])&(r[1:-1]>r[2:])&ph[1:-1])[0]+1
    ks=sorted([peri[2],apo[apo>peri[2]][0],peri[3],apo[apo>peri[3]][0]])
    fig,ax=plt.subplots(1,2,figsize=(14,5)); draw(ax[0],ax[1],F,ks,xlim=(2,8),ylim=(-3,3),cmin=t[ks[0]]-30,cmax=t[ks[-1]]+30)
    save(fig,'frozen_ueff_rocking.pdf')

    # ---- 4. bound excursion on the fractal side (the contact member of the neighbor pair) ----
    d=np.load(data('cA2_12.npz')); o,n,b=d['out'],d['npk'],d['b']
    cand=[i for i in range(len(b)-1) if o[i]!=o[i+1] and min(n[i],n[i+1])>=12]; i=cand[len(cand)//2]
    P=[dense(b[i],mA,npt=10),dense(b[i+1],mA,npt=10)]; tr=[p for p in P if p['outcome']=='contact'][0]
    F=frozen(tr); r,t=F['r'],F['t']; print('excursion chk',F['chk'])
    kap=int(np.argmax(np.where(t>350,r,0)))
    ks=[tidx(F,(t>250)&(r>6)),kap,tidx(F,(t>t[kap])&(r<6)),len(t)-1]
    fig,ax=plt.subplots(1,2,figsize=(14,5)); draw(ax[0],ax[1],F,ks,xlim=(2,13),ylim=(-1.5,2.5),cmin=250,cmax=t[-1])
    save(fig,'frozen_ueff_bound_excursion.pdf')

"""Slide figures (PDF) for the APS EGLS 2026 talk, built from the upper-edge chaos data.
Needs data/: bd_s025.npz, cB0_12.npz, cA0_12.npz, cA1_12.npz, cA2_12.npz, stairB_1e-13.npz, unc_d.json.
Writes to output/ (or UEC_OUT)."""
from paths import data, out
import numpy as np, json, os, pickle, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle
from probe import mirror, I, E0
from physics.solve import run_trajectory

plt.rcParams.update({'font.size':14,'axes.titlesize':15,'axes.labelsize':14,'legend.fontsize':12,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#666',
                     'pdf.fonttype':42,'savefig.bbox':'tight'})
CB,CE,INK,MUTE,C1,C2='#2a78d6','#eb6834','#222','#999','#2a78d6','#c0392b'
mB=mirror(np.pi/4,-0.07*np.pi); mA=mirror(np.pi/4,-0.042*np.pi)
be=float(np.load(data('stairB_1e-13.npz'))['be'])

def dense(b,m,rt=1e-12,rcut=14,tail=40,npt=25):
    """Full 18-component state on a uniform grid while the pair is interacting."""
    cache=data(f'dense_{b:.12f}_{m[1][0]:.6f}_{rt:.0e}_{rcut}_{tail}_{npt}.pkl')
    if os.path.exists(cache): return pickle.load(open(cache,'rb'))
    r=run_trajectory(E0,b,m[0],m[1],start_radius=64000.,stop='escape',rtol=rt,atol=rt*1e-2,method='reference',keep_solution=True)
    sol=r['solution']; rr=np.linalg.norm(sol.y[0:3],axis=0); k=np.where(rr<rcut)[0]
    t0,t1=sol.t[k[0]],min(sol.t[-1],sol.t[k[-1]]+tail)
    tt=np.linspace(t0,t1,int((t1-t0)*npt)+2); Y=sol.sol(tt)
    res=dict(b=b,outcome=r['outcome'],t=tt-tt[0],y=Y)
    pickle.dump(res,open(cache,'wb')); return res

def angles(y):
    """Moment angles relative to the line of centers (r = r2 - r1), in degrees."""
    th=np.arctan2(y[1],y[0])
    w=lambda a:np.degrees((a-th+np.pi)%(2*np.pi)-np.pi)
    return w(np.arctan2(y[7],y[6])),w(np.arctan2(y[10],y[9]))

def krot(y): return 0.5*I*((y[12:15]**2).sum(0)+(y[15:18]**2).sum(0))

def save(fig,name):
    fig.savefig(out(name)); plt.close(fig); print('wrote',name)

# ---------------- Slide 6: band overview (the tongue) ----------------
f=np.load(data('bd_s025.npz')); o,n,ds,bs=f['out'],f['npk'],f['ds'],f['bs']
img=np.where(o==1,-1,np.minimum(n,4))
cmap=ListedColormap(['#dfe9f7','#fbd5c2','#f29a70','#d9531e','#8a2c08'])
norm=BoundaryNorm([-1.5,-0.5,1.5,2.5,3.5,4.5],cmap.N)
fig,ax=plt.subplots(figsize=(10,5.4))
im=ax.pcolormesh(bs,ds/np.pi,img,cmap=cmap,norm=norm,shading='nearest',rasterized=True)
cb=fig.colorbar(im,ax=ax,ticks=[-1,1,2,3,4]); cb.ax.set_yticklabels(['contact','1','2','3','≥ 4']); cb.set_label('periapses before escape')
ax.axhline(-0.07,color=INK,lw=1,ls='--'); ax.text(5.9,-0.0685,'B',ha='right',fontsize=12)
ax.axhline(-0.042,color=INK,lw=1,ls='--'); ax.text(5.9,-0.0405,'A',ha='right',fontsize=12)
for b0,lab in [(3.0,'a'),(3.2,'b'),(3.4,'c')]:
    ax.plot(b0,-0.07,'o',ms=7,mfc='white',mec=INK); ax.text(b0,-0.0755,lab,ha='center',va='top',fontsize=12)
ax.set_xlabel('impact parameter  b / R'); ax.set_ylabel('offset from mirror line  d / π')
save(fig,'band_overview_bd.pdf')

# ---------------- Slide 6: three trajectories ----------------
cases=[(3.0,'a'),(3.2,'b'),(3.4,'c')]
fig,axs=plt.subplots(1,3,figsize=(15,4.8))
for ax,(b0,title) in zip(axs,cases):
    tr=dense(b0,mB,rt=1e-12,rcut=12,tail=30,npt=20); y=tr['y']; t=tr['t']; r=y[0:3]; rr=np.linalg.norm(r,axis=0)
    p1,p2=-r[:2]/2,r[:2]/2
    ax.plot(*p1,color=C1,lw=1,alpha=0.6); ax.plot(*p2,color=C2,lw=1,alpha=0.6)
    esc=tr['outcome']=='escape'; kmin=int(np.argmin(rr)) if esc else len(t)-1
    kin=int(np.where(rr[:kmin]>7)[0][-1]); snaps=[(kin,0.35,'in')]
    if esc:
        w=np.where(rr[kmin:]>7)[0]; snaps.append((kmin+int(w[0]) if len(w) else len(t)-1,0.35,'out'))
    snaps.append((kmin,1.0,'closest approach' if esc else 'contact'))
    for k,al,lab in snaps:
        lw=2.4 if al==1 else 1.3
        for p,M,c in [(p1[:,k],y[6:8,k],C1),(p2[:,k],y[9:11,k],C2)]:
            u_=M/np.linalg.norm(M)
            ax.add_patch(Circle(p,1,fc='white' if al==1 else 'none',ec=c,lw=lw,alpha=al,zorder=3))
            ax.annotate('',xy=p+0.8*u_,xytext=p-0.8*u_,arrowprops=dict(arrowstyle='-|>',color=c,lw=lw,alpha=al,mutation_scale=16),zorder=4)
    ax.set_aspect('equal'); ax.set_xlim(-6.5,6.5); ax.set_ylim(-3.6,3.6); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    res='escape' if esc else 'contact'
    ax.set_title(f'({title})  b = {b0}R:  {res}',fontsize=15)
save(fig,'window_trajectories.pdf')

# ---------------- Slide 7: staircase ----------------
sB=np.load(data('stairB_1e-13.npz')); dx,so,sn=sB['dx'],sB['out'],sB['npk']
fig,ax=plt.subplots(figsize=(8,5.4))
ax.semilogx(dx[so==0],sn[so==0],'o',ms=4,color=CE)
k=np.arange(2,15,2); x0=dx[(so==0)&(sn==4)].max()
ax.semilogx(x0*24.0**(-(k-4)/2),k,'+',ms=12,mew=1.5,color=INK,label='×1/24 per step')
ax.invert_xaxis(); ax.set_xlabel('distance below the edge,  b_edge − b   (R)'); ax.set_ylabel('periapses before escape')
ax.legend(frameon=False,loc='upper left'); ax.grid(alpha=.2,which='both')
save(fig,'staircase.pdf')

# ---------------- Slide 7: rocking orbit ----------------
tr=dense(be-3e-9,mB,rt=1e-13); y=tr['y']; t=tr['t']; rr=np.linalg.norm(y[0:3],axis=0); a1,a2=angles(y)
fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[1,1]})
ax[0].plot(t,rr,color=INK,lw=1.3); ax[0].axhline(2,color=MUTE,ls='--',lw=1)
ax[0].set_ylim(0,10); ax[0].set_ylabel('r / R')
ax[1].plot(t,a1,color=C1,lw=1.2,label='magnet 1'); ax[1].plot(t,a2,color=C2,lw=1.2,label='magnet 2')
ax[1].axhspan(-50,50,color='#000',alpha=0.05,lw=0); ax[1].set_ylabel('moment angle from r̂ (°)'); ax[1].set_yticks([-180,-90,0,90,180])
ax[1].set_xlabel('time (t₀)'); ax[1].legend(frameon=False,ncol=2,loc='lower right')
for a in ax: a.grid(alpha=.2)
save(fig,'rocking_orbit.pdf')

# ---------------- Slide 8: zooms (regular vs fractal) ----------------
def strip(ax,fn,title,span=None,b0=None,unit=1.0):
    d=np.load(data(fn)); b,o,n=d['b'],d['out'],d['npk']; e=o==0
    if b0 is not None:
        b=(b-b0)/unit; ax.set_xlabel(f'(b − {b0}) / {unit:.0e}  (R)'.replace('e-0','e-'))
        if span: span=((span[0]-b0)/unit,(span[1]-b0)/unit)
    else: ax.set_xlabel('b / R')
    ax.scatter(b[e],n[e],s=4,color=CE,lw=0,rasterized=True,label='escape')
    top=max(n[e].max() if e.any() else 1,6)
    ax.vlines(b[o==1],-0.14*top,-0.04*top,color=CB,lw=0.6,rasterized=True,label='contact')
    ax.set_ylim(-0.17*top,1.08*top); ax.set_title(title,loc='left',fontsize=13); ax.grid(alpha=.15)
    if span: ax.axvspan(*span,color='#000',alpha=0.08,lw=0)
    ax.ticklabel_format(useOffset=True,axis='x')
fig,axs=plt.subplots(2,2,figsize=(14,7.2))
strip(axs[0,0],'cB0_12.npz','regular, d = −0.070π')
axs[0,0].legend(frameon=False,fontsize=11,loc='upper left')
strip(axs[0,1],'cA0_12.npz','fractal, d = −0.042π',span=(1.8975,1.9025))
strip(axs[1,0],'cA1_12.npz','×16',span=(1.89925,1.89945),b0=1.8975,unit=1e-3)
strip(axs[1,1],'cA2_12.npz','×400',b0=1.89925,unit=1e-5)
for a in axs[:,0]: a.set_ylabel('periapses')
fig.tight_layout(); save(fig,'fractal_zooms.pdf')

# ---------------- Slide 8: neighbor pair ----------------
d=np.load(data('cA2_12.npz')); o,n,b=d['out'],d['npk'],d['b']
cand=[i for i in range(len(b)-1) if o[i]!=o[i+1] and min(n[i],n[i+1])>=12]; i=cand[len(cand)//2]
P=[dense(b[i],mA,npt=10),dense(b[i+1],mA,npt=10)]
fig,ax=plt.subplots(figsize=(11,4.8))
for tr,c in zip(P,[CE,CB]):
    ax.plot(tr['t'],np.linalg.norm(tr['y'][0:3],axis=0),color=c,lw=1.0,label=tr['outcome'])
ax.axhline(2,color=MUTE,ls='--',lw=1); ax.set_ylim(0,min(30,max(np.linalg.norm(tr['y'][0:3],axis=0).max() for tr in P))*1.1)
ax.set_xlabel('time (t₀)'); ax.set_ylabel('r / R'); ax.legend(frameon=False,loc='upper left')
ax.grid(alpha=.2)
save(fig,'neighbor_pair.pdf')

# ---------------- Slide 8: uncertain fraction across d ----------------
u=json.load(open(data('unc_d.json')))
fig,ax=plt.subplots(figsize=(8.5,5.4))
for r in u:
    e=np.array(r['eps']); f_=np.array(r['counts'])/r['n']; k=f_>0
    frac=r['dpi']>-0.052; c=plt.cm.Oranges(0.45+0.5*(r['dpi']+0.052)/0.007) if frac else plt.cm.Blues(0.45+0.5*(-0.052-r['dpi'])/0.016)
    lab=f"{r['dpi']:.3f}π"
    ax.loglog(e[k],f_[k],'o-',color=c,ms=5,lw=1.4,label=lab)
ee=np.array([3e-6,3e-4]); ax.loglog(ee,0.012*ee/3e-4,color=INK,ls='--',lw=1); ax.text(1.5e-5,2e-4,'slope 1',fontsize=12)
ax.set_xlabel('ε / R'); ax.set_ylabel('uncertain fraction  f(ε)'); ax.grid(alpha=.2,which='both')
ax.legend(frameon=False,fontsize=11,loc='center left',bbox_to_anchor=(1.01,0.5),title='d')
save(fig,'uncertainty_vs_d.pdf')

# ---------------- Slide 10: spin as energy reservoir ----------------
tr=[p for p in P if p['outcome']=='contact'][0]; y=tr['y']; t=tr['t']; rr=np.linalg.norm(y[0:3],axis=0); Kr=krot(y)/E0
Eorb=1-Kr
fig,ax=plt.subplots(2,1,figsize=(11,6.2),sharex=True)
ax[0].plot(t,rr,color=INK,lw=1); ax[0].axhline(2,color=MUTE,ls='--',lw=1); ax[0].set_ylabel('r / R'); ax[0].set_ylim(0,min(30,rr.max())*1.1)
ax[1].plot(t,Kr,color=CE,lw=1); ax[1].axhline(1,color=INK,lw=1)
far=(rr>8)&(t>t[np.argmax(rr<5)]); ka=np.where(far)[0][np.argmax(rr[far])]; kf=Kr[ka]
for a in ax: a.fill_between(t,0,1,where=far,transform=a.get_xaxis_transform(),color=CE,alpha=0.12,lw=0)
ax[0].text(t[far].mean(),3,f'K_rot = {kf:.2f} E',ha='center',fontsize=12)
ax[1].set_ylabel('K_rot / E'); ax[1].set_xlabel('time (t₀)')
for a in ax: a.grid(alpha=.2)
save(fig,'spin_reservoir.pdf')

# ---------------- Slide 7: rocking orbit in space ----------------
tr=dense(be-3e-9,mB,rt=1e-13); y=tr['y']; t=tr['t']; r=y[0:3]; rr=np.linalg.norm(r,axis=0)
p1,p2=-r[:2]/2,r[:2]/2
ph=(t>50)&(t<190)
peri=np.where((rr[1:-1]<rr[:-2])&(rr[1:-1]<rr[2:])&ph[1:-1])[0]+1
ts=np.linspace(t[peri[2]],t[peri[4]],6)[:-1]    # one rocking period, 5 snapshots
fig,axs=plt.subplots(1,5,figsize=(16,3.8))
for ax,tk in zip(axs,ts):
    k=int(np.argmin(abs(t-tk)))
    ax.plot(*p1[:,ph],color=C1,lw=0.6,alpha=0.25); ax.plot(*p2[:,ph],color=C2,lw=0.6,alpha=0.25)
    for p,M,c in [(p1[:,k],y[6:8,k],C1),(p2[:,k],y[9:11,k],C2)]:
        u_=M/np.linalg.norm(M)
        ax.add_patch(Circle(p,1,fc='white',ec=c,lw=2.2,zorder=3))
        ax.annotate('',xy=p+0.8*u_,xytext=p-0.8*u_,arrowprops=dict(arrowstyle='-|>',color=c,lw=2.2,mutation_scale=16),zorder=4)
    ax.plot([p1[0,k],p2[0,k]],[p1[1,k],p2[1,k]],color=MUTE,lw=1,ls=':',zorder=2)
    ax.set_title(f't = {tk-ts[0]:.0f} t₀',fontsize=14)
    ax.set_aspect('equal'); L=3.4; ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
save(fig,'rocking_orbit_trajectory.pdf')

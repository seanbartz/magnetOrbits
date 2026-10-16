from paths import data, OUT
import os
import numpy as np, json, pickle, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#888',
                     'xtick.color':'#555','ytick.color':'#555','axes.titlesize':10})
CB,CE,INK,MUTE='#2a78d6','#eb6834','#222','#999'
OUT=OUT+os.sep
def strip(ax,fn,title,spans=()):
    d=np.load(fn); b,o,n=d['b'],d['out'],d['npk']; e=o==0
    ax.scatter(b[e],n[e],s=2.5,color=CE,lw=0,label='escape (y = periapses)',rasterized=True)
    top=max(n[e].max() if e.any() else 1,6)
    ax.vlines(b[o==1],-0.12*top,-0.03*top,color=CB,lw=0.6,label='contact',rasterized=True)
    ax.set_ylim(-0.15*top,1.08*top)
    for (a,c) in spans: ax.axvspan(a,c,color='#000',alpha=0.07,lw=0)
    ax.set_title(title,loc='left'); ax.set_ylabel('periapses before escape'); ax.grid(alpha=.15)
    ax.ticklabel_format(useOffset=True,axis='x'); return d
# ---------- Figure 1: overview ----------
f=np.load(data('bd_s025.npz')); out,npk,ds,bs=f['out'],f['npk'],f['ds'],f['bs']
img=np.where(out==1,-1,np.minimum(npk,6))
cmap=ListedColormap(['#dfe9f7','#fbe0d3','#f2a07a','#e0602a','#b33d12','#7a2408','#4a1504'])
norm=BoundaryNorm([-1.5,-0.5,1.5,2.5,3.5,4.5,5.5,6.5],cmap.N)
fig,ax=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1.5,1]})
im=ax[0].pcolormesh(bs,ds/np.pi,img,cmap=cmap,norm=norm,shading='nearest',rasterized=True)
cb=fig.colorbar(im,ax=ax[0],ticks=[-1,1,2,3,4,5,6]); cb.ax.set_yticklabels(['contact','escape, 1','2','3','4','5','≥6']); cb.set_label('outcome / periapses before escape')
for dv,lab in [(-0.070,'cut B (regular edge)'),(-0.042,'cut A (fractal edge)')]:
    ax[0].axhline(dv,color=INK,lw=0.8,ls='--'); ax[0].text(5.95,dv+0.0015,lab,ha='right',fontsize=8,color=INK)
ax[0].axhline(-0.052,color=MUTE,lw=0.8,ls=':'); ax[0].text(0.55,-0.0505,'onset of fractal edges ≈ −0.052π',fontsize=7.5,color='#555')
ax[0].set_xlabel('b / R'); ax[0].set_ylabel('d / π   (offset below the mirror line φ₁+φ₂ = π)')
ax[0].set_title('Mirror band at s = π/4 (φ₁ = π/4, φ₂ = 3π/4 + d), Ẽ = 0.0335: the escape window is the thin tongue',loc='left',fontsize=9)
u=json.load(open(data('unc_d.json'))); u2=json.load(open(data('unc.json')))
blues=plt.cm.Greys(np.linspace(0.35,0.95,len(u)))
for c,r in zip(blues,u):
    e=np.array(r['eps']); n=np.array(r['counts'])/4000.; k=n>0
    ax[1].loglog(e[k],n[k],'o-',color=c,ms=3.5,lw=1.2,label=f"d = {r['dpi']:.3f}π"+(f"  α = {r['alpha']:.2f}" if r['alpha']==r['alpha'] else ''))
ee=np.array([1e-7,3e-4]); ax[1].loglog(ee,0.02*(ee/3e-4),color=CB,lw=1,ls='--'); ax[1].text(2e-6,1.5e-4,'slope 1 (finite set of smooth edges)',color=CB,fontsize=7.5,rotation=27)
ax[1].set_xlabel('ε'); ax[1].set_ylabel('uncertain fraction f(ε)'); ax[1].legend(fontsize=7,frameon=False,loc='lower right')
ax[1].set_title('Uncertainty exponent across the band (4000 samples per d)',loc='left',fontsize=9); ax[1].grid(alpha=.15,which='both')
fig.tight_layout(); fig.savefig(OUT+'chaos_1_band_overview.png',dpi=140); plt.close(fig)
# ---------- Figure 2: regular vs fractal edge ----------
fig,axs=plt.subplots(3,2,figsize=(13,10))
strip(axs[0,0],data('cB0_12.npz'),'B (d = −0.070π): window and both edges')
axs[0,0].legend(frameon=False,fontsize=8,loc='upper center')
sB=np.load(data('stairB_1e-13.npz')); dx,o,n=sB['dx'],sB['out'],sB['npk']
a=axs[1,0]; a.semilogx(dx,n,'o',ms=2.5,color=CE)
k=np.arange(2,15,2); lam=24.0; x0=1.18e-3
a.semilogx(x0*lam**(-(k-4)/2),k,'+',color=INK,ms=7,label='geometric, λ = 24 per 2 periapses')
a.invert_xaxis(); a.set_xlabel('distance below the upper edge, b_edge − b'); a.set_ylabel('periapses before escape')
a.set_title('B: approach to the upper edge is a clean staircase (rtol 10⁻¹³)',loc='left'); a.legend(frameon=False,fontsize=8); a.grid(alpha=.15,which='both')
T=pickle.load(open(data('trajs.pkl'),'rb'))
a=axs[2,0]
for tr,c in zip(T['B'],plt.cm.Oranges(np.linspace(0.35,1,6))):
    a.plot(tr['t'],np.linalg.norm(tr['y'][0:3],axis=0),color=c,lw=1.2,label=f"b_edge − b = {T['B'][0]['b']*0+ (sB['be']-tr['b']):.0e}")
tr=T['B'][-1]; a.plot(tr['t'],np.linalg.norm(tr['y'][0:3],axis=0),color=CB,lw=1.5,label='b_edge + 1e-4 (contact)')
a.axhline(2,color=MUTE,lw=.8,ls='--'); a.set_ylim(0,12); a.set_xlabel('time (t₀) after r first drops below 12R'); a.set_ylabel('r / R')
a.set_title('B: each step closer to the edge buys one more rocking cycle at r ≈ 3.3R',loc='left'); a.legend(frameon=False,fontsize=7,ncol=2); a.grid(alpha=.15)
d=strip(axs[0,1],data('cA0_12.npz'),'A (d = −0.042π): window; both edges are fuzzy',spans=[(1.8975,1.9025)])
strip(axs[1,1],data('cA1_12.npz'),'A, zoom ×16 into the upper edge (shaded above)',spans=[(1.89925,1.89945)])
strip(axs[2,1],data('cA2_12.npz'),'A, zoom ×400: a smaller copy of the window, again with fuzzy edges')
axs[2,1].set_xlabel('b / R')
for a in (axs[0,0],axs[0,1],axs[1,1]): a.set_xlabel('b / R')
fig.suptitle('Two kinds of window edge. B: the edge is the stable manifold of one unstable orbiting state. A: windows nest inside windows (fractal, D ≈ 0.72).',fontsize=10)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT+'chaos_2_regular_vs_fractal_edge.png',dpi=140); plt.close(fig)
# ---------- Figure 3: trapped orbits ----------
fig,axs=plt.subplots(1,3,figsize=(13,4.3),gridspec_kw={'width_ratios':[1,1,1.5]})
for a,tr,title in [(axs[0],T['B'][5],'B, 3e-9 below the edge: rocking orbit'),(axs[1],T['A'][1],'A: long chaotic transient')]:
    x,y=tr['y'][0],tr['y'][1]; a.plot(x,y,color=CE,lw=0.6); a.add_patch(plt.Circle((0,0),2,fc='#eee',ec=MUTE))
    a.set_aspect('equal'); a.set_xlim(-7,7); a.set_ylim(-7,7); a.set_title(title,loc='left'); a.set_xlabel('x / R (relative)'); a.grid(alpha=.15)
axs[0].set_ylabel('y / R'); axs[0].text(0,0,'r < 2\n(contact)',ha='center',va='center',fontsize=7,color='#555')
a=axs[2]
for tr,c in zip(T['A'],[CB,CE]):
    a.plot(tr['t'],np.linalg.norm(tr['y'][0:3],axis=0),color=c,lw=0.8,label=f"b = {tr['b']:.10f} → {tr['outcome']}")
rm=max(np.linalg.norm(tr['y'][0:3],axis=0).max() for tr in T['A']); a.axhline(2,color=MUTE,lw=.8,ls='--'); a.set_ylim(0,min(rm,60)*1.15); a.set_xlabel('time (t₀)'); a.set_ylabel('r / R'); a.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0.08,1.0))
a.set_title('A: two impact parameters 2.5×10⁻⁸ apart', loc='left'); a.grid(alpha=.15)
fig.tight_layout(); fig.savefig(OUT+'chaos_3_trapped_orbits.png',dpi=140); plt.close(fig)

"""Animated GIFs of the frozen-U_eff picture: space view (center-of-mass frame, moments),
instantaneous U_eff with the system at (r, E - K_rot), and the path in (r, E - K_rot).
Usage: python talk_ueff_gifs.py [name ...]   (default: all). Writes to output/."""
import sys, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrow
from paths import data, out
from talk_ueff_frozen import dense, frozen, CMAP, E0, mB, mA, be

C1,C2='#2a78d6','#c0392b'

def orbits():
    d=np.load(data('cA2_12.npz')); o,n,b=d['out'],d['npk'],d['b']
    cand=[i for i in range(len(b)-1) if o[i]!=o[i+1] and min(n[i],n[i+1])>=12]; i=cand[len(cand)//2]
    P=[dense(b[i],mA,npt=10),dense(b[i+1],mA,npt=10)]
    return {
      'head_to_tail':  dict(tr=dense(5.0,(np.array([1.,0,0]),np.array([1.,0,0])),rcut=12,tail=30,npt=40),xlim=12,ylim=(-2,3),L=7,nf=160),
      'window_b3.0':   dict(tr=dense(3.0,mB,rcut=12,tail=30,npt=20),xlim=10,ylim=(-3,4),L=6,nf=160),
      'window_b3.2':   dict(tr=dense(3.2,mB,rcut=12,tail=30,npt=20),xlim=10,ylim=(-3,4),L=6,nf=160),
      'window_b3.4':   dict(tr=dense(3.4,mB,rcut=12,tail=30,npt=20),xlim=10,ylim=(-3,4),L=6,nf=160),
      'rocking':       dict(tr=dense(be-3e-9,mB,rt=1e-13),xlim=10,ylim=(-3,3),L=6,nf=320),
      'bound_excursion':dict(tr=[p for p in P if p['outcome']=='contact'][0],xlim=13,ylim=(-1.5,2.5),L=8,nf=400),
    }

def make(name,cfg):
    tr=cfg['tr']; F=frozen(tr); y=tr['y']; t=F['t']; r=F['r']
    p1,p2=-y[0:2]/2,y[0:2]/2
    idx=np.linspace(0,len(t)-1,cfg['nf']).astype(int)
    rg=np.linspace(2,cfg['xlim'],400); ylo,yhi=cfg['ylim']; T=t[-1]
    fig,ax=plt.subplots(1,3,figsize=(16,5.8),gridspec_kw={'width_ratios':[1,1.15,1.15]})
    a0,a1,a2=ax
    L=cfg['L']; a0.set_xlim(-L,L); a0.set_ylim(-L,L); a0.set_aspect('equal'); a0.set_xticks([]); a0.set_yticks([])
    for s in a0.spines.values(): s.set_visible(False)
    for a in (a1,a2):
        a.axvspan(1.6,2,color='#000',alpha=0.08,lw=0); a.axhline(0,color='#bbb',lw=0.8)
        a.set_xlim(1.6,cfg['xlim']); a.set_ylim(ylo,yhi); a.set_xlabel(r'$r/R$')
        for s in ('top','right'): a.spines[s].set_visible(False)
    a1.set_ylabel(r'$U_{\rm eff}/E$'); a2.set_ylabel(r'$(E-K_{\rm rot})/E$')
    a2.plot(r,F['Erad'],color='#e6e6e6',lw=0.8)
    tr1,=a0.plot([],[],color=C1,lw=1,alpha=0.5); tr2,=a0.plot([],[],color=C2,lw=1,alpha=0.5)
    s1=Circle((0,0),1,fc='white',ec=C1,lw=2); s2=Circle((0,0),1,fc='white',ec=C2,lw=2); a0.add_patch(s1); a0.add_patch(s2)
    arr=[]
    cur,=a1.plot([],[],color='#222',lw=2); dot1,=a1.plot([],[],'o',ms=10,mec='white',mew=1.2)
    path=a2.scatter([np.nan],[np.nan],c=[0.],s=5,linewidths=0,edgecolors='none',cmap=CMAP,vmin=0,vmax=T); dot2,=a2.plot([],[],'o',ms=10,mec='white',mew=1.2)
    tt=a0.text(0,-L*0.95,'',ha='center',fontsize=19,color='#444')
    def frame(j):
        k=idx[j]; col=CMAP(t[k]/T)
        w=slice(max(0,k-int(len(t)*0.15)),k+1)
        tr1.set_data(*p1[:,w]); tr2.set_data(*p2[:,w]); s1.center=p1[:,k]; s2.center=p2[:,k]
        for a_ in arr: a_.remove()
        arr.clear()
        for p,M,c in [(p1[:,k],y[6:8,k],C1),(p2[:,k],y[9:11,k],C2)]:
            u=M/np.linalg.norm(M)*0.8
            arr.append(a0.add_patch(FancyArrow(*(p-u),*(2*u),width=0.09,head_width=0.38,head_length=0.38,length_includes_head=True,color=c,zorder=5)))
        cur.set_data(rg,(F['L'][k]**2/(2*rg**2)-F['C'][k]/rg**3)/E0); cur.set_color(col)
        dot1.set_data([r[k]],[F['Erad'][k]]); dot1.set_color(col); dot1.set_markeredgecolor('white')
        sel=np.arange(0,k+1,max(1,k//1500+1))
        path.set_offsets(np.c_[r[sel],F['Erad'][sel]]); path.set_array(t[sel])
        dot2.set_data([r[k]],[F['Erad'][k]]); dot2.set_color(col); dot2.set_markeredgecolor('white')
        tt.set_text(rf'$t={t[k]:.0f}\,t_0$')
        return []
    fig.tight_layout()
    anim=FuncAnimation(fig,frame,frames=len(idx),blit=False)
    anim.save(out(f'frozen_ueff_{name}.gif'),writer=PillowWriter(fps=20),dpi=72)
    plt.close(fig); print('wrote',name,len(idx),'frames')

if __name__=='__main__':
    O=orbits(); names=sys.argv[1:] or list(O)
    for n in names: make(n,O[n])

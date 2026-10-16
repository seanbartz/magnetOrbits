from paths import data
import numpy as np, sys, json
from multiprocessing import Pool
from probe import probe, mirror
from physics.fractal import sample_points, uncertain_counts, fit_exponent, effective_length
EPS=[3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7]
f0=np.load(data('bd_s025.npz')); out=f0['out']; ds=f0['ds']; bs=f0['bs']
code={0:'e',1:'c',2:'t'}
G={}
def f(b): return code[probe(b,*G['m'],rtol=1e-12,atol=1e-14)[0]]
def init(m): G['m']=m
if __name__=='__main__':
    res=[]
    for dpi in [-0.068,-0.064,-0.060,-0.057,-0.054,-0.051,-0.048,-0.045]:
        i=np.argmin(abs(ds/np.pi-dpi)); dd=ds[i]
        tr=np.where(np.diff(out[i])!=0)[0]
        if len(tr)<2: print('d/pi=%.4f: no window in bd_s025 row, skipped'%(dd/np.pi),flush=True); continue
        lo,hi=bs[tr[0]]-0.005,bs[tr[1]+1]+0.005
        m=mirror(np.pi/4,dd); n=4000
        c,eps,ball=sample_points(lo,hi,EPS,n,seed=11)
        with Pool(2,initializer=init,initargs=(m,)) as p: codes=''.join(p.map(f,ball,chunksize=500))
        rows=uncertain_counts(codes,n,eps); cnt=np.array([r[0] for r in rows]); tot=np.array([r[1] for r in rows])
        L=effective_length(lo,hi,EPS); a,ae,use=fit_exponent(eps,cnt,tot,min_count=10,eps_max=L/30)
        res.append(dict(dpi=dd/np.pi,b=[lo,hi],n=n,alpha=a,alpha_err=ae,counts=cnt.tolist(),eps=list(eps)))
        print('d/pi=%.4f b=[%.3f,%.3f] alpha=%.3f±%.3f'%(dd/np.pi,lo,hi,a,ae),cnt.tolist(),flush=True)
    json.dump(res,open(data('unc_d.json'),'w'),indent=1)

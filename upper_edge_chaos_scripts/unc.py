import numpy as np, sys, json
from multiprocessing import Pool
sys.path.insert(0,'scratch')
from probe import probe, mirror
from physics.fractal import sample_points, uncertain_counts, fit_exponent, effective_length
EPS=[1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7]
cases={'A_chaotic':(-0.042,1.8975,1.9025),'B_regular':(-0.070,3.10,3.30)}
code={0:'e',1:'c',2:'t'}
M={}
def f(args):
    k,b=args; m1,m2=M[k]; return code[probe(b,m1,m2,rtol=1e-12,atol=1e-14)[0]]
for k,(dpi,lo,hi) in cases.items(): M[k]=mirror(np.pi/4,dpi*np.pi)
if __name__=='__main__':
    res={}
    for k,(dpi,lo,hi) in cases.items():
        n=12000
        c,eps,ball=sample_points(lo,hi,EPS,n,seed=7)
        with Pool(2) as p: codes=''.join(p.map(f,[(k,b) for b in ball],chunksize=500))
        rows=uncertain_counts(codes,n,eps)
        cnt=np.array([r[0] for r in rows]); tot=np.array([r[1] for r in rows])
        L=effective_length(lo,hi,EPS)
        a,ae,use=fit_exponent(eps,cnt,tot,min_count=10,eps_max=L/30)
        res[k]=dict(dpi=dpi,b=[lo,hi],eps=list(eps),counts=cnt.tolist(),totals=tot.tolist(),alpha=a,alpha_err=ae,used=use.tolist(),L=L)
        print(k,'alpha=%.3f±%.3f'%(a,ae),'counts',cnt.tolist())
    json.dump(res,open('scratch/unc.json','w'),indent=1)

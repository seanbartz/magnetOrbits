import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
fig,axs=plt.subplots(len(sys.argv)-2,1,figsize=(13,2.6*(len(sys.argv)-2)))
for ax,fn in zip(np.atleast_1d(axs),sys.argv[2:]):
    f=np.load(fn); b,o,n=f['b'],f['out'],f['npk']
    ax.scatter(b,np.where(o==1,-1,n),c=np.where(o==1,'C0','C3'),s=2); ax.set_title(fn); ax.set_ylabel('npk (escape) / -1 contact')
fig.tight_layout(); fig.savefig(sys.argv[1],dpi=80)

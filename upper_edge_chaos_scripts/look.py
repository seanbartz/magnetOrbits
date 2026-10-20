import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
f=np.load(sys.argv[1]); out,npk,ds,bs=f['out'],f['npk'],f['ds'],f['bs']
fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].pcolormesh(bs,ds/np.pi,out,cmap='coolwarm_r',shading='nearest'); ax[0].set_title('outcome (red=contact? check)'); ax[0].set_xlabel('b'); ax[0].set_ylabel('d/pi')
im=ax[1].pcolormesh(bs,ds/np.pi,np.where(out==1,-1,npk),cmap='viridis',vmin=-1,vmax=6,shading='nearest'); fig.colorbar(im,ax=ax[1]); ax[1].set_title('periapses before escape (-1=contact)')
fig.savefig(sys.argv[2],dpi=90)

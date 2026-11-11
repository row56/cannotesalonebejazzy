import sys, json, numpy as np
sys.path.insert(0,'.')
from melodicvae.style import auc
e=np.load('subaxes/out/enc.npz',allow_pickle=True); a=np.load('subaxes/out/axes.npz')
z,D,names,g=e['z'],e['D'],list(e['names']),e['genre']
col=lambda n:D[:,names.index(n)]
pJ=(z-a['mean'])@a['J']
X=np.c_[np.ones(len(z)),col('syncopation'),col('lhl_syncopation'),col('chromaticism'),col('note_density'),col('pc_entropy')]
res=pJ-X@np.linalg.lstsq(X,pJ,rcond=None)[0]
print('axis AUC raw',round(auc(pJ[g==1],pJ[g==0]),3),'| after regressing out sync,lhl,chroma,density,pc_entropy',round(auc(res[g==1],res[g==0]),3))
# stratified: bins on lhl sync & chroma quantiles, weighted within-bin AUC
qs=lambda v,k: np.digitize(v,np.quantile(v,np.linspace(0,1,k+1)[1:-1]))
strata=qs(col('lhl_syncopation'),5)*100+qs(col('chromaticism')+1e-9*np.random.rand(len(z)),3)*10+qs(col('note_density'),3)
num=den=0
for s in np.unique(strata):
    m=strata==s; nj,nc=(g[m]==1).sum(),(g[m]==0).sum()
    if nj>=10 and nc>=10:
        w=nj*nc; num+=w*auc(pJ[m&(g==1)],pJ[m&(g==0)]); den+=w
print('within-stratum AUC (matched on lhl sync x chroma x density)',round(num/den,3))
json.dump({'auc_raw':round(auc(pJ[g==1],pJ[g==0]),3),'auc_resid':round(auc(res[g==1],res[g==0]),3),'auc_stratified':round(num/den,3)},open('subaxes/out/control.json','w'))

import sys, json, numpy as np
sys.path.insert(0,'.')
from melodicvae.style import StyleSpace, auc, _unit
e=np.load('subaxes/out/enc.npz',allow_pickle=True)
z,D,names,g,corp,cn=e['z'],e['D'],list(e['names']),e['genre'],e['corpus'],list(e['cn'])
sp=StyleSpace.load('./runs/base_v6/axes_dm_heldout/styles.npz')
J=sp.contrast_axis('jaz','jazz','classical'); mean=sp.mean
def ridge(v):
    zc=z-mean; v=(v-v.mean())/(v.std()+1e-8)
    return _unit(np.linalg.solve(zc.T@zc+1e-3*len(zc)*np.eye(128),zc.T@v))
def perp(a,bs):
    for b in bs: a=a-(a@b)*b
    return _unit(a)
col=lambda n:D[:,names.index(n)]
Dn=ridge(col('note_density'))
S=perp(ridge(col('lhl_syncopation')),[Dn])
C=perp(ridge(col('chromaticism')),[Dn,S])
R=perp(J,[Dn,S,C])
ax={'jazz (paper)':J,'syncopation':S,'chromaticism':C,'jazz minus sync/chroma/density':R}
desc=['note_density','syncopation','lhl_syncopation','chromaticism']
out={}
for k,a in ax.items():
    p=(z-mean)@a
    r={n:round(float(np.corrcoef(p,col(n))[0,1]),2) for n in desc}
    a_all=auc(p[g==1],p[g==0])
    # format-controlled: score cells (pdmx vs lieder), performance cells (jtd vs asap)
    ci={n:i for i,n in enumerate(cn)}
    a_sc=auc(p[corp==ci['pdmx_jazz']],p[corp==ci['lieder']])
    a_pf=auc(p[corp==ci['jtd']],p[corp==ci['asap']])
    out[k]={'corr':r,'genre_auc':round(a_all,3),'auc_scores':round(a_sc,3),'auc_performances':round(a_pf,3),'sd':float(p.std()),'cos_with_jazz':round(float(a@J),2)}
    print(k,json.dumps(out[k]))
# baseline: descriptors alone (sync+chroma+density) linear, in-sample AUC via logistic-ish: use LDA direction
X=np.c_[col('lhl_syncopation'),col('chromaticism'),col('note_density')]; X=(X-X.mean(0))/X.std(0)
w=np.linalg.solve(np.cov(X.T),X[g==1].mean(0)-X[g==0].mean(0)); s=X@w
print('descriptor trio LDA AUC',round(auc(s[g==1],s[g==0]),3))
out['_descriptor_trio_auc']=round(auc(s[g==1],s[g==0]),3)
np.savez('subaxes/out/axes.npz',J=J,S=S,C=C,R=R,Dn=Dn,mean=mean)
json.dump(out,open('subaxes/out/axes_report.json','w'),indent=1)

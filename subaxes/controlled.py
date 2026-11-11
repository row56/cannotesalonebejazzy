import sys, json, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'./scripts')
from melodicvae.style import auc, _unit
from melodicvae.repr import ONSET_OFFSET
from melodicvae.attributes import describe
from melodicvae.model import MelodyVAE, VAEConfig
from build_demo import sampling_bias
e=np.load('subaxes/out/enc.npz',allow_pickle=True); a=np.load('subaxes/out/axes.npz')
x=np.load('./data/processed/train_v6.npz',allow_pickle=True)['x'].astype(np.int64)
z,D,names,g=e['z'],e['D'],list(e['names']),e['genre']; col=lambda n:D[:,names.index(n)]
rng=np.random.default_rng(0)
qs=lambda v,k: np.digitize(v,np.quantile(v,np.linspace(0,1,k+1)[1:-1]))
st=qs(col('syncopation')+1e-9*rng.random(len(z)),4)*1000+qs(col('lhl_syncopation')+1e-9*rng.random(len(z)),5)*100+qs(col('chromaticism')+1e-9*rng.random(len(z)),3)*10+qs(col('note_density'),4)
J_idx,C_idx=[],[]
for s in np.unique(st):
    j=np.where((st==s)&(g==1))[0]; c=np.where((st==s)&(g==0))[0]; n=min(len(j),len(c))
    if n: J_idx+=list(rng.choice(j,n,replace=False)); C_idx+=list(rng.choice(c,n,replace=False))
J_idx,C_idx=np.array(J_idx),np.array(C_idx)
bal=lambda n:(round(float(col(n)[J_idx].mean()),3),round(float(col(n)[C_idx].mean()),3))
print('matched sample',len(J_idx),'per genre; lhl',bal('lhl_syncopation'),'chroma',bal('chromaticism'),'dens',bal('note_density'))
M=_unit(z[J_idx].mean(0)-z[C_idx].mean(0))
p=(z-a['mean'])@M
rep={'n_matched_per_genre':int(len(J_idx)),'cos_with_paper_axis':round(float(M@a['J']),2),
 'auc_all':round(auc(p[g==1],p[g==0]),3),'auc_matched':round(auc(p[J_idx],p[C_idx]),3),
 'corr':{n:round(float(np.corrcoef(p,col(n))[0,1]),2) for n in ('lhl_syncopation','syncopation','chromaticism','note_density')}}
print(rep)
np.save('subaxes/out/matched_axis.npy',M)
b=torch.load('./runs/base_v6/best.pt',map_location='cpu',weights_only=False)
m=MelodyVAE(VAEConfig(**b['cfg'].get('model',{}))); m.load_state_dict(b['model']); m.eval()
SDJ=float(((z-a['mean'])@a['J']).std())
nn=(x>=ONSET_OFFSET).sum(1); r7=np.random.default_rng(7)
idx=np.r_[r7.choice(np.where((g==0)&(nn>=8))[0],10,replace=False),r7.choice(np.where((g==1)&(nn>=8))[0],10,replace=False)]
K=['lhl_syncopation','syncopation','chromaticism','note_density','rest_ratio','beat_vacancy','mean_interval','pitch_range','pc_entropy']
out={}
for k,ax in (('paper axis',a['J']),('matched axis',M)):
    r={'-3':[],'+3':[]}
    for i in idx:
        t=x[i]; bias=sampling_bias(t,0.0,12,'cpu'); torch.manual_seed(int(i))
        for s,al in (('-3',-3),('+3',3)):
            zz=torch.tensor(np.repeat((z[i]+al*SDJ*ax)[None],48,0),dtype=torch.float32)
            with torch.no_grad(): c=m.generate(zz,64,temperature=0.9,top_p=0.98,logit_bias=bias).numpy()
            cnt=(c>=ONSET_OFFSET).sum(1); c=c[int(np.argmin(abs(cnt-nn[i])))]
            d=describe(c); r[s].append([d[q] for q in K])
    lo,hi=np.mean(r['-3'],0),np.mean(r['+3'],0)
    out[k]={q:[round(float(l),3),round(float(h),3)] for q,l,h in zip(K,lo,hi)}; print(k,out[k])
rep['steer_20']=out
json.dump(rep,open('subaxes/out/matched_report.json','w'),indent=1)

import sys, json, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'./scripts')
from melodicvae.style import auc, _unit
from melodicvae.repr import ONSET_OFFSET, tokens_to_midi
from melodicvae.attributes import describe
from melodicvae.model import MelodyVAE, VAEConfig
from build_demo import sampling_bias
BLUE={3,6,10}  # b3 b5 b7
def blue(t):
    p=t[t>=ONSET_OFFSET]-ONSET_OFFSET
    return float(np.isin(p%12,list(BLUE)).mean()) if len(p) else 0.0
e=np.load('subaxes/out/enc.npz',allow_pickle=True); a=np.load('subaxes/out/axes.npz')
x=np.load('data/processed/train_v6.npz',allow_pickle=True)['x'].astype(np.int64)
z,D,names,g,corp,cn=e['z'],e['D'],list(e['names']),e['genre'],e['corpus'],list(e['cn'])
B=np.array([blue(t) for t in x]); mean=a['mean']
zc=z-mean; v=(B-B.mean())/B.std()
w=np.linalg.solve(zc.T@zc+1e-3*len(zc)*np.eye(128),zc.T@v); w/=np.linalg.norm(w)
for b in (a['Dn'],a['S']): w=w-(w@b)*b
BL=w/np.linalg.norm(w)
p=zc@BL; ci={n:i for i,n in enumerate(cn)}
rep={'blue_rate_jazz':round(float(B[g==1].mean()),3),'blue_rate_classical':round(float(B[g==0].mean()),3),
 'descriptor_auc':round(auc(B[g==1],B[g==0]),3),'axis_auc':round(auc(p[g==1],p[g==0]),3),
 'axis_auc_scores':round(auc(p[corp==ci['pdmx_jazz']],p[corp==ci['lieder']]),3),
 'axis_auc_perf':round(auc(p[corp==ci['jtd']],p[corp==ci['asap']]),3),
 'corr':{n:round(float(np.corrcoef(p,D[:,names.index(n)])[0,1]),2) for n in ('note_density','lhl_syncopation','chromaticism')}|{'blue':round(float(np.corrcoef(p,B)[0,1]),2)},
 'cos_jazz':round(float(BL@a['J']),2)}
print(rep)
b=torch.load('runs/base_v6/best.pt',map_location='cpu',weights_only=False)
m=MelodyVAE(VAEConfig(**b['cfg'].get('model',{}))); m.load_state_dict(b['model']); m.eval()
SDJ=float((zc@a['J']).std())
rng=np.random.default_rng(7); nn=(x>=ONSET_OFFSET).sum(1)
idx=np.r_[rng.choice(np.where((g==0)&(nn>=8))[0],10,replace=False),rng.choice(np.where((g==1)&(nn>=8))[0],10,replace=False)]
def dec(zv,t,bias,seed):
    torch.manual_seed(seed); zz=torch.tensor(np.repeat(zv[None],48,0),dtype=torch.float32)
    with torch.no_grad(): c=m.generate(zz,64,temperature=0.9,top_p=0.98,logit_bias=bias).numpy()
    cnt=(c>=ONSET_OFFSET).sum(1); return c[int(np.argmin(abs(cnt-(t>=ONSET_OFFSET).sum())))]
st={}
for k,ax in (('blues',BL),('jazz',a['J'])):
    r={'-3':[],'+3':[]}
    for i in idx:
        t=x[i]; bias=sampling_bias(t,0.0,12,'cpu')
        for s,al in (('-3',-3),('+3',3)):
            c=dec(z[i]+al*SDJ*ax,t,bias,int(i)); d=describe(c); r[s].append([blue(c),d['chromaticism'],d['lhl_syncopation'],d['note_density']])
    lo,hi=np.mean(r['-3'],0),np.mean(r['+3'],0)
    st[k]={n:[round(float(l),3),round(float(h),3)] for n,l,h in zip(('blue','chromaticism','lhl_syncopation','note_density'),lo,hi)}
    print(k,st[k])
rep['steer_20']=st
np.save('subaxes/out/blues_axis.npy',BL)
json.dump(rep,open('subaxes/out/blues_report.json','w'),indent=1)

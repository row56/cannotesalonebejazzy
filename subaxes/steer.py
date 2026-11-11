import sys, json, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'./scripts')
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.repr import ONSET_OFFSET
from melodicvae.attributes import describe
from build_demo import sampling_bias
b=torch.load('./runs/base_v6/best.pt',map_location='cpu',weights_only=False)
m=MelodyVAE(VAEConfig(**b['cfg'].get('model',{}))); m.load_state_dict(b['model']); m.eval()
e=np.load('subaxes/out/enc.npz',allow_pickle=True); a=np.load('subaxes/out/axes.npz')
x=np.load('./data/processed/train_v6.npz',allow_pickle=True)['x'].astype(np.int64)
z,g=e['z'],e['genre']
rng=np.random.default_rng(7)
nn=(x>=ONSET_OFFSET).sum(1)
idx=np.r_[rng.choice(np.where((g==0)&(nn>=8))[0],10,replace=False),rng.choice(np.where((g==1)&(nn>=8))[0],10,replace=False)]
AX={'syncopation':a['S'],'chromaticism':a['C'],'jazz':a['J'],'jazz_residual':a['R']}
SDJ=float(((z-a['mean'])@a['J']).std()); SD={k:SDJ for k in AX}
KEYS=['lhl_syncopation','syncopation','chromaticism','note_density']
def decode(zv,t,bias,n=48,seed=0):
    torch.manual_seed(seed)
    zz=torch.tensor(np.repeat(zv[None],n,0),dtype=torch.float32)
    with torch.no_grad(): c=m.generate(zz,64,temperature=0.9,top_p=0.98,logit_bias=bias).numpy()
    tgt=int((t>=ONSET_OFFSET).sum()); cnt=(c>=ONSET_OFFSET).sum(1)
    return c[int(np.argmin(abs(cnt-tgt)))]
res={k:{'-3':[], '+3':[]} for k in AX}
for i in idx:
    t=x[i]; bias=sampling_bias(t,0.0,12,'cpu')
    for k,v in AX.items():
        for s,al in (('-3',-3),('+3',3)):
            c=decode(z[i]+al*SD[k]*v,t,bias,seed=int(i))
            res[k][s].append([describe(c)[q] for q in KEYS])
out={}
for k in AX:
    lo,hi=np.array(res[k]['-3']),np.array(res[k]['+3'])
    out[k]={q:[round(float(lo[:,j].mean()),3),round(float(hi[:,j].mean()),3)] for j,q in enumerate(KEYS)}
    print(k,out[k])
json.dump({'sd':SD,'n':len(idx),'means_minus3_plus3':out},open('subaxes/out/steer_equal.json','w'),indent=1)

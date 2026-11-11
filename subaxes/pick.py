import sys,json,numpy as np,torch
sys.path.insert(0,'.'); sys.path.insert(0,'./scripts')
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.repr import ONSET_OFFSET, REST, HOLD
from melodicvae.attributes import describe
from build_demo import sampling_bias
b=torch.load('runs/base_v6/best.pt',map_location='cpu',weights_only=False)
m=MelodyVAE(VAEConfig(**b['cfg'].get('model',{}))); m.load_state_dict(b['model']); m.eval()
a=np.load('subaxes/out/axes.npz'); e=np.load('subaxes/out/enc.npz',allow_pickle=True); z,g=e['z'],e['genre']
x=np.load('data/processed/train_v6.npz',allow_pickle=True); X=x['x'].astype(np.int64); src=x['source_file']
p=(z-a['mean'])@a['J']; mid=(p[g==1].mean()+p[g==0].mean())/2; SDJ=float(p.std())
BL=np.load('subaxes/out/blues_axis.npy')
nn=(X>=ONSET_OFFSET).sum(1)
def blue(c):
    q=c[c>=ONSET_OFFSET]-ONSET_OFFSET; return float(np.isin(q%12,[3,6,10]).mean())
cand=[]
print('pool', int(((abs(p-mid)<0.35*SDJ)&(nn>=12)&(nn<=22)).sum()))
for i in np.where((abs(p-mid)<0.35*SDJ)&(nn>=12)&(nn<=22))[0]:
    d=describe(X[i]); t=X[i]
    durs=[]; k=0
    on=np.where(t>=ONSET_OFFSET)[0]
    if len(set(np.diff(on)))<3: continue          # want rhythmic variety
    if d['pitch_range']<6 or d['rest_ratio']>0.2: continue
    q=t[t>=ONSET_OFFSET]-ONSET_OFFSET; oos=int((~np.isin(q%12,[0,2,4,5,7,9,11])).sum())
    if oos>1: continue
    if not (0.15<=d['lhl_syncopation']<=0.8): continue
    cand.append(i)
rng=np.random.default_rng(1); cand=rng.choice(cand,min(80,len(cand)),replace=False)
res=[]
def gen(z0,ax,al,t,bias):
    torch.manual_seed(1000); zz=torch.tensor(np.repeat((z0+al*SDJ*ax)[None],64,0),dtype=torch.float32)
    with torch.no_grad(): c=m.generate(zz,64,temperature=0.9,top_p=0.98,logit_bias=bias).numpy()
    cnt=(c>=ONSET_OFFSET).sum(1); return c[int(np.argmin(abs(cnt-(t>=ONSET_OFFSET).sum())))]
for i in cand:
    t=X[i]; bias=sampling_bias(t,0.0,12,'cpu'); z0=z[i]
    d0=describe(t)
    J3=describe(gen(z0,a['J'],3,t,bias)); Jm=describe(gen(z0,a['J'],-3,t,bias))
    S=describe(gen(z0,a['S'],6,t,bias)); Sm=describe(gen(z0,a['S'],-6,t,bias))
    C=gen(z0,a['C'],6,t,bias); Cd=describe(C)
    B=gen(z0,BL,3,t,bias)
    score=(J3['lhl_syncopation']-Jm['lhl_syncopation'])+(S['lhl_syncopation']-Sm['lhl_syncopation'])-abs(S['chromaticism']-d0['chromaticism'])*3 \
          +(Cd['chromaticism']-d0['chromaticism'])*4+(blue(B)-blue(t))*4+(J3['lhl_syncopation']-d0['lhl_syncopation'])
    res.append((round(float(score),2),int(i),str(src[i]),round(float((p[i]-mid)/SDJ),2),int(nn[i]),round(float(d0['lhl_syncopation']),2),round(float(d0['syncopation']),2)))
res.sort(reverse=True)
for r in res[:10]: print(r)

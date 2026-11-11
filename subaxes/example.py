import sys,json,numpy as np,torch,subprocess
sys.path.insert(0,'.'); sys.path.insert(0,'./scripts')
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.repr import tokens_to_midi, ONSET_OFFSET
from melodicvae.attributes import describe
from build_demo import sampling_bias
I=int(sys.argv[1]); P=sys.argv[2]
b=torch.load('runs/base_v6/best.pt',map_location='cpu',weights_only=False)
m=MelodyVAE(VAEConfig(**b['cfg'].get('model',{}))); m.load_state_dict(b['model']); m.eval()
a=np.load('subaxes/out/axes.npz'); e=np.load('subaxes/out/enc.npz'); z=e['z']; SDJ=float(((z-a['mean'])@a['J']).std())
BL=np.load('subaxes/out/blues_axis.npy')
X=np.load('data/processed/train_v6.npz',allow_pickle=True)['x'].astype(np.int64)
t=X[I]; bias=sampling_bias(t,0.0,12,'cpu'); n=int((t>=ONSET_OFFSET).sum()); z0=z[I]
NM=['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
def info(c):
    d=describe(c); q=c[c>=ONSET_OFFSET]-ONSET_OFFSET
    return {k:round(float(d[k]),2) for k in ('lhl_syncopation','syncopation','chromaticism','note_density')}|{'blue':round(float(np.isin(q%12,[3,6,10]).mean()),2),'notes':' '.join(f'{NM[v%12]}@{s}' for s,v in zip(np.where(c>=ONSET_OFFSET)[0],q))}
def out(c,lab):
    f=f'subaxes/out/audio/{P}_{lab}.mid'; tokens_to_midi(c,grid_hz=80/60*4).write(f)
    subprocess.run(['fluidsynth','-ni','-F',f[:-4]+'.wav','-r','44100','/usr/share/sounds/sf2/default-GM.sf2',f],capture_output=True)
out(t,'orig'); rep={'orig':info(t)}; print('orig',rep['orig'])
for lab,ax,al in (('jazz_m3',a['J'],-3),('jazz_p3',a['J'],3),('sync_p6',a['S'],6),('sync_m6',a['S'],-6),('chroma_p6',a['C'],6),('blues_p3',BL,3),('blues_p6',BL,6)):
    torch.manual_seed(1000); zz=torch.tensor(np.repeat((z0+al*SDJ*ax)[None],96,0),dtype=torch.float32)
    with torch.no_grad(): c=m.generate(zz,64,temperature=0.9,top_p=0.98,logit_bias=bias).numpy()
    cnt=(c>=ONSET_OFFSET).sum(1); c=c[int(np.argmin(abs(cnt-n)))]
    out(c,lab); rep[lab]=info(c); print(lab,rep[lab])
json.dump(rep,open(f'subaxes/out/{P}_report.json','w'),indent=1)

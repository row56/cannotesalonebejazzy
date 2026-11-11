import sys, json, numpy as np, torch, subprocess
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'./scripts')
src=open('./scripts/steer_tunes.py').read()
ns={}; exec(src.split('SIGMA, T')[0].replace('R = sys.argv[1]; OUT = Path(sys.argv[2]); sys.path.insert(0, R)','R="."'),ns)
tok,TUNES,sampling_bias=ns['tok'],ns['TUNES'],ns['sampling_bias']
from melodicvae.model import MelodyVAE, VAEConfig
b=torch.load('./runs/base_v6/best.pt',map_location='cpu',weights_only=False)
m=MelodyVAE(VAEConfig(**b['cfg'].get('model',{}))); m.load_state_dict(b['model']); m.eval()
from melodicvae.repr import ONSET_OFFSET, tokens_to_midi
from melodicvae.attributes import describe
a=np.load('subaxes/out/axes.npz'); e=np.load('subaxes/out/enc.npz')
SDJ=float(((e['z']-a['mean'])@a['J']).std())
AL=int(sys.argv[1]); OUT=Path('subaxes/out/audio'); OUT.mkdir(exist_ok=True)
grid_hz=80/60*4; rep={}
for name,seq in TUNES.items():
    t=tok(seq).astype(np.int64); bias=sampling_bias(t,0.0,12,'cpu'); tgt=int((t>=ONSET_OFFSET).sum())
    with torch.no_grad(): z0=m.encode(torch.tensor(t)[None])[0].numpy()[0]
    rep[name]={}
    for ax in ('S','C'):
        for al in (-AL,AL):
            torch.manual_seed(1000)
            zz=torch.tensor(np.repeat((z0+al*SDJ*a[ax])[None],96,0),dtype=torch.float32)
            with torch.no_grad(): c=m.generate(zz,64,temperature=0.9,top_p=0.98,logit_bias=bias).numpy()
            cnt=(c>=ONSET_OFFSET).sum(1); c=c[int(np.argmin(abs(cnt-tgt)))]
            lab=f"{name}_{'sync' if ax=='S' else 'chroma'}_{'m' if al<0 else 'p'}{AL}"
            tokens_to_midi(c,grid_hz=grid_hz).write(str(OUT/f'{lab}.mid'))
            rep[name][lab]={k:round(float(v),3) for k,v in describe(c).items() if k in('lhl_syncopation','chromaticism','note_density')}
            rep[name][lab]['notes']=int((c>=ONSET_OFFSET).sum())
print(json.dumps(rep,indent=1)); json.dump(rep,open(OUT/'tunes.json','w'),indent=1)
for f in OUT.glob('*.mid'):
    subprocess.run(['fluidsynth','-ni','-F',str(f.with_suffix('.wav')),'-r','44100','/usr/share/sounds/sf2/default-GM.sf2',str(f)],capture_output=True)

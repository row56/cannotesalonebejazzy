import sys, numpy as np, torch
sys.path.insert(0,'.')
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.attributes import describe_batch, DESCRIPTORS
b=torch.load('runs/base_v6/best.pt',map_location='cpu',weights_only=False)
m=MelodyVAE(VAEConfig(**b['cfg'].get('model',{}))); m.load_state_dict(b['model']); m.eval()
d=np.load('data/processed/train_v6.npz',allow_pickle=True)
x=torch.tensor(d['x'].astype(np.int64))
with torch.no_grad():
    zs=[m.encode(x[i:i+512])[0].numpy() for i in range(0,len(x),512)]
z=np.concatenate(zs); D=describe_batch(d['x'])
np.savez('subaxes/out/enc.npz',z=z,D=D,names=np.array(list(DESCRIPTORS)),genre=d['label_genre'],gn=d['label_genre_names'],corpus=d['label_corpus'],cn=d['label_corpus_names'])
print(z.shape,D.shape,list(DESCRIPTORS),d['label_genre_names'],d['label_corpus_names'])

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.data import MelodyDataset
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.style import encode_dataset


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--styles', required=True)
    ap.add_argument('--axis', default='jazz')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = pick_device()
    blob = torch.load(args.ckpt, map_location=device)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {})))
    model.load_state_dict(blob['model'])
    ds = MelodyDataset(args.data, augment_transpose=0)
    z = encode_dataset(model, DataLoader(ds, batch_size=256), device)
    print(f'encoded {len(z)} windows into {z.shape[1]}-d latents')
    st = np.load(args.styles, allow_pickle=True)
    names = [str(n) for n in st['axis_names']]
    if args.axis not in names:
        raise SystemExit(f'unknown axis {args.axis!r}; have {names}')
    axis = st['axis_vectors'][names.index(args.axis)]
    proj = z @ axis
    np.savez_compressed(
        args.out,
        projection=proj.astype(np.float32),
        axis_name=np.array(args.axis),
        corpus=ds.codes('corpus'),
        corpus_names=np.array(ds.names('corpus'), dtype=object),
        genre=ds.codes('genre'),
        genre_names=np.array(ds.names('genre'), dtype=object),
    )
    print(f'wrote {args.out}: {len(proj)} projections onto {args.axis!r}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import describe
from melodicvae.data import MelodyDataset
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.repr import tokens_to_midi
from melodicvae.style import StyleSpace
from melodicvae.train import pick_device


def render(model, z: np.ndarray, out: Path, name: str, device: str, temperature: float, bpm: float, steps: int):
    zt = torch.tensor(np.atleast_2d(z), dtype=torch.float32, device=device)
    tokens = model.generate(zt, steps=steps, temperature=temperature).cpu().numpy()[0]
    grid_hz = bpm / 60.0 * 4
    tokens_to_midi(tokens, grid_hz=grid_hz).write(str(out / f'{name}.mid'))
    d = describe(tokens)
    print(f'  {name:<32} ' + '  '.join((f'{k}={v:.2f}' for k, v in d.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--styles', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('demos'))
    ap.add_argument('--axis', default=None, help='axis name to traverse')
    ap.add_argument('--seed-index', type=int, default=0, help='which window to use as seed')
    ap.add_argument('--alphas', type=float, nargs='*', default=[-3, -1.5, 0, 1.5, 3])
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--bpm', type=float, default=110)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    # print(args)
    torch.manual_seed(args.seed)
    device = pick_device()
    blob = torch.load(args.ckpt, map_location=device)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {}))).to(device)
    model.load_state_dict(blob['model'])
    model.eval()
    space = StyleSpace.load(args.styles)
    ds = MelodyDataset(args.data, augment_transpose=0)
    steps = ds.x.shape[1]
    args.out.mkdir(parents=True, exist_ok=True)
    seed_tokens = ds.x[args.seed_index]
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(seed_tokens.astype(np.int64), device=device).unsqueeze(0))
    z0 = mu.cpu().numpy()[0]
    tokens_to_midi(seed_tokens, grid_hz=args.bpm / 60 * 4).write(str(args.out / 'seed.mid'))
    tags = ', '.join((f'{f}={ds.names(f)[int(ds.codes(f)[args.seed_index])]}' for f in ds.fields))
    print(f'seed: window {args.seed_index}  ({tags})')
    if args.axis not in space.axes:
        raise SystemExit(f'unknown axis {args.axis!r}; have {list(space.axes)}')
    for a in args.alphas:
        render(
            model,
            space.steer(z0, args.axis, a),
            args.out,
            f'{args.axis}_a{a:+.1f}'.replace('.', 'p'),
            device,
            args.temperature,
            args.bpm,
            steps,
        )
    print(f'\nwrote midi to {args.out}')


if __name__ == '__main__':
    main()

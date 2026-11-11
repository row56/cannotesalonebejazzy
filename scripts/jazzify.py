#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import describe
from melodicvae.data import MelodyDataset, midi_to_tokens
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.repr import ONSET_OFFSET, STEPS_PER_BAR, tokens_to_midi
from melodicvae.style import StyleSpace, encode_dataset
from melodicvae.train import pick_device


def concat_midi(clips, grid_hz, out_path, gap_steps=0):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    step = 1.0 / grid_hz
    offset = 0.0
    for tokens in clips:
        part = tokens_to_midi(tokens, grid_hz=grid_hz)
        for src in part.instruments:
            for n in src.notes:
                inst.notes.append(
                    pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start + offset, end=n.end + offset)
                )
        offset += (len(tokens) + gap_steps) * step
    pm.instruments.append(inst)
    pm.write(str(out_path))
    return offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--midi', type=Path, required=True, help='melody to jazzify')
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--styles', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True, help='corpus the projection spread is measured on')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--field', default='genre')
    ap.add_argument('--positive', default='jazz')
    ap.add_argument('--negative', default='classical')
    ap.add_argument('--axis-name', default='jazziness')
    ap.add_argument('--alpha-lo', type=float, default=0.0)
    ap.add_argument('--alpha-hi', type=float, default=3.0)
    ap.add_argument('--steps', type=int, default=7, help='slider positions between lo and hi, inclusive')
    ap.add_argument(
        '--window',
        type=int,
        default=0,
        help="which 4-bar window of the input to use; -1 sweeps every window and keeps the melody's full length",
    )
    ap.add_argument('--bars', type=int, default=2, help='bars per clip; capped at the training window')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.92)
    ap.add_argument('--bpm', type=float, default=60)
    ap.add_argument('--gap-beats', type=float, default=1.0, help='silence between positions in the continuous file')
    ap.add_argument('--sigma-sample', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    # args.alphas = [-3, 3]
    tokens_full = midi_to_tokens(args.midi)
    if tokens_full is None:
        raise SystemExit(f'could not read a melody out of {args.midi}')
    onsets = int((tokens_full >= ONSET_OFFSET).sum())
    print(f'{args.midi.name}: {len(tokens_full)} steps, {onsets} onsets after skyline')
    device = pick_device()
    blob = torch.load(args.ckpt, map_location=device)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {}))).to(device)
    model.load_state_dict(blob['model'])
    model.eval()
    ds = MelodyDataset(args.data)
    seq_len = ds.x.shape[1]
    clip_steps = min(args.bars * STEPS_PER_BAR, seq_len)
    grid_hz = args.bpm / 60.0 * 4
    if len(tokens_full) < seq_len:
        raise SystemExit(
            f"the melody is {len(tokens_full)} steps, shorter than the model's {seq_len}-step window ({seq_len // STEPS_PER_BAR} bars). Give it more bars."
        )
    starts = (
        list(range(0, len(tokens_full) - seq_len + 1, clip_steps)) if args.window < 0 else [args.window * clip_steps]
    )
    if starts[-1] + seq_len > len(tokens_full):
        starts = [s for s in starts if s + seq_len <= len(tokens_full)]
    if not starts:
        raise SystemExit('no usable window at that offset')
    space = StyleSpace.load(args.styles)
    axis = space.contrast_axis(args.axis_name, args.positive, args.negative)
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(ds), size=min(args.sigma_sample, len(ds)), replace=False)
    sub = torch.utils.data.Subset(ds, sample.tolist())
    sigma = float((encode_dataset(model, DataLoader(sub, batch_size=256), device) @ axis).std())
    print(f'axis {args.axis_name!r}: sigma={sigma:.3f}')
    alphas = np.linspace(args.alpha_lo, args.alpha_hi, args.steps) if args.steps > 1 else np.array([args.alpha_hi])
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'input': str(args.midi),
        'axis': args.axis_name,
        'sigma': round(sigma, 4),
        'alphas': [round(float(a), 3) for a in alphas],
        'bpm': args.bpm,
        'bars': args.bars,
        'clip_steps': clip_steps,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'windows': len(starts),
        'clips': [],
    }
    ramp = []
    for w, start in enumerate(starts):
        seed_tokens = tokens_full[start : start + seq_len]
        tokens_to_midi(seed_tokens[:clip_steps], grid_hz=grid_hz).write(str(args.out / f'w{w:02d}_original.mid'))
        with torch.no_grad():
            mu, _ = model.encode(torch.tensor(seed_tokens.astype(np.int64), device=device).unsqueeze(0))
        z0 = mu.cpu().numpy()[0]
        for k, alpha in enumerate(alphas):
            z = torch.tensor(np.atleast_2d(z0 + float(alpha) * sigma * axis), dtype=torch.float32, device=device)
            torch.manual_seed(args.seed * 1000 + w)
            out_tokens = (
                model.generate(z, steps=clip_steps, temperature=args.temperature, top_p=args.top_p).cpu().numpy()[0]
            )
            name = f'w{w:02d}_s{k:02d}_a{alpha:+.2f}'.replace('.', 'p')
            tokens_to_midi(out_tokens, grid_hz=grid_hz).write(str(args.out / f'{name}.mid'))
            ramp.append(out_tokens)
            d = describe(out_tokens)
            manifest['clips'].append(
                {
                    'window': w,
                    'step': k,
                    'alpha': round(float(alpha), 3),
                    'file': f'{name}.mid',
                    'descriptors': {kk: round(float(vv), 4) for kk, vv in d.items()},
                }
            )
            print(
                f"  w{w:02d} a{alpha:+.2f}  notes={int(d['note_density'] * clip_steps):3d} chroma={d['chromaticism']:.2f}"
            )
    gap = int(round(args.gap_beats * 4))
    total = concat_midi(ramp, grid_hz, args.out / 'ramp.mid', gap_steps=gap)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nwrote {len(ramp)} clips and ramp.mid ({total:.1f}s) to {args.out}')


if __name__ == '__main__':
    main()

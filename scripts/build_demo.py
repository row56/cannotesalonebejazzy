#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.attributes import best_scale, chromaticism, describe, leap_rate, mean_interval, pitch_range
from melodicvae.data import MelodyDataset
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.repr import ONSET_OFFSET, STEPS_PER_BAR, VOCAB_SIZE, tokens_to_midi
from melodicvae.style import StyleSpace, encode_dataset
from melodicvae.train import pick_device


def onset_counts(x: np.ndarray) -> np.ndarray:
    return (x >= 2).sum(1)


def seed_centre(tokens: np.ndarray) -> int:
    idx = np.asarray(tokens).ravel()
    pitches = idx[idx >= ONSET_OFFSET] - ONSET_OFFSET
    return int(np.median(pitches)) if len(pitches) else 60


def sampling_bias(tokens: np.ndarray, key_penalty: float, register_span: int, device) -> torch.Tensor:
    bias = torch.zeros(VOCAB_SIZE, device=device)
    scale = set((int(pc) for pc in best_scale(tokens)[0]))
    centre = seed_centre(tokens)
    lo = centre - register_span // 2
    hi = lo + register_span - 1
    for tok in range(ONSET_OFFSET, VOCAB_SIZE):
        pitch = tok - ONSET_OFFSET
        if register_span > 0 and (not lo <= pitch <= hi):
            bias[tok] = float('-inf')
        elif key_penalty > 0 and pitch % 12 not in scale:
            bias[tok] = -key_penalty
    return bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--styles', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--field', default='genre')
    ap.add_argument('--positive', default='jazz')
    ap.add_argument('--negative', default='classical')
    ap.add_argument('--axis-name', default='jazziness')
    ap.add_argument('--per-side', type=int, default=6, help='seed melodies drawn from each side of the axis')
    ap.add_argument(
        '--alphas',
        type=float,
        nargs='*',
        default=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
        help='slider positions, in corpus standard deviations',
    )
    ap.add_argument('--sigma-sample', type=int, default=20000, help='windows encoded to estimate the projection spread')
    ap.add_argument('--bars', type=int, default=2, help='bars per clip; capped at the training window (4 bars)')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.92, help='nucleus cutoff; 1.0 samples the full distribution')
    ap.add_argument(
        '--key-bias', type=float, default=2.5, help="logit penalty on notes outside the seed's scale; 0 disables"
    )
    ap.add_argument(
        '--max-seed-chromaticism',
        type=float,
        default=0.12,
        help='reject seed melodies with more out-of-scale notes than this',
    )
    ap.add_argument(
        '--register-span',
        type=int,
        default=12,
        help="semitones the decoder may use, centred on the seed's median pitch; 0 lets it use the full range",
    )
    ap.add_argument(
        '--max-seed-range', type=float, default=14, help='reject seed melodies spanning more semitones than this'
    )
    ap.add_argument(
        '--max-seed-mean-interval',
        type=float,
        default=0,
        help='reject seeds whose mean absolute interval exceeds this, in semitones; 0 disables. Hand-transcribed melody sits near 2.8, a piano skyline near 6-8',
    )
    ap.add_argument(
        '--max-seed-leap-rate',
        type=float,
        default=0,
        help='reject seeds where more than this share of intervals exceeds a fifth; 0 disables. Real melody sits near 0.04',
    )
    ap.add_argument(
        '--match-candidates',
        type=int,
        default=1,
        help='draws per slider position to pick a density-matched clip from; 1 keeps the single shared-rng draw',
    )
    ap.add_argument('--match-tol', type=int, default=3, help='onset-count spread across a row still called matched')
    ap.add_argument('--bpm', type=float, default=110)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = pick_device()
    blob = torch.load(args.ckpt, map_location=device)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {}))).to(device)
    model.load_state_dict(blob['model'])
    model.eval()
    ds = MelodyDataset(args.data, augment_transpose=0)
    steps = ds.x.shape[1]
    clip_steps = min(args.bars * STEPS_PER_BAR, steps)
    grid_hz = args.bpm / 60.0 * 4
    # grid_hz = 120 / 60.0 * 4
    space = StyleSpace.load(args.styles)
    axis = space.contrast_axis(args.axis_name, args.positive, args.negative)
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(ds), size=min(args.sigma_sample, len(ds)), replace=False)
    sub = torch.utils.data.Subset(ds, sample.tolist())
    z_sample = encode_dataset(model, DataLoader(sub, batch_size=256), device)
    sigma = float((z_sample @ axis).std())
    print(f'axis {args.axis_name!r}: projection sigma={sigma:.3f} over {len(sample)} windows')
    counts = onset_counts(ds.x)
    lo, hi = np.percentile(counts, [40, 90])
    eligible = (counts >= lo) & (counts <= hi)
    if args.max_seed_chromaticism < 1.0:
        chroma = np.array([chromaticism(w) for w in ds.x])
        eligible &= chroma <= args.max_seed_chromaticism
        print(f'{eligible.sum()} windows eligible after the chromaticism filter (<= {args.max_seed_chromaticism})')
    if args.max_seed_range > 0:
        spans = np.array([pitch_range(w) for w in ds.x])
        eligible &= spans <= args.max_seed_range
        print(f'{eligible.sum()} windows eligible after the range filter (<= {args.max_seed_range:g} semitones)')
    if args.max_seed_mean_interval > 0:
        mi = np.array([mean_interval(w) for w in ds.x])
        eligible &= mi <= args.max_seed_mean_interval
        print(
            f'{eligible.sum()} windows eligible after the mean-interval filter (<= {args.max_seed_mean_interval:g} semitones)'
        )
    if args.max_seed_leap_rate > 0:
        lr = np.array([leap_rate(w) for w in ds.x])
        eligible &= lr <= args.max_seed_leap_rate
        print(f'{eligible.sum()} windows eligible after the leap-rate filter (<= {args.max_seed_leap_rate:g})')
    names = ds.names(args.field)
    codes = ds.codes(args.field)
    picks = []
    for side in (args.negative, args.positive):
        pool = np.flatnonzero(eligible & (codes == names.index(side)))
        picks.extend(((side, int(i)) for i in rng.choice(pool, args.per_side, replace=False)))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'axis': args.axis_name,
        'positive': args.positive,
        'negative': args.negative,
        'sigma': round(sigma, 4),
        'bpm': args.bpm,
        'bars': args.bars,
        'clip_steps': clip_steps,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'key_bias': args.key_bias,
        'register_span': args.register_span,
        'max_seed_mean_interval': args.max_seed_mean_interval,
        'max_seed_leap_rate': args.max_seed_leap_rate,
        'match_candidates': args.match_candidates,
        'match_tol': args.match_tol,
        'alphas': args.alphas,
        'seeds': [],
    }
    for rank, (side, index) in enumerate(picks):
        tokens = ds.x[index]
        with torch.no_grad():
            mu, _ = model.encode(torch.tensor(tokens.astype(np.int64), device=device).unsqueeze(0))
        z0 = mu.cpu().numpy()[0]
        stem = f'seed{rank:02d}'
        (args.out / stem).mkdir(exist_ok=True)
        tokens_to_midi(tokens[:clip_steps], grid_hz=grid_hz).write(str(args.out / stem / 'original.mid'))
        bias = sampling_bias(tokens, args.key_bias, args.register_span, device)
        source = str(ds.source_file[index]) if ds.source_file is not None else ''
        entry = {
            'id': stem,
            'index': index,
            'side': side,
            'source_file': source,
            'original': f'{stem}/original.mid',
            'clips': [],
        }

        def sample(alpha, rng_seed):
            z = z0 + alpha * sigma * axis
            zt = torch.tensor(np.atleast_2d(z), dtype=torch.float32, device=device)
            torch.manual_seed(rng_seed)
            return (
                model.generate(zt, steps=clip_steps, temperature=args.temperature, top_p=args.top_p, logit_bias=bias)
                .cpu()
                .numpy()[0]
            )

        base_rng = args.seed * 1000 + rank
        if args.match_candidates > 1:
            target = int((tokens[:clip_steps] >= ONSET_OFFSET).sum())
            chosen, spread = ({}, [])
            for alpha in args.alphas:
                cands = [sample(alpha, base_rng * 100 + j) for j in range(args.match_candidates)]
                counts = [int((c >= ONSET_OFFSET).sum()) for c in cands]
                best = int(np.argmin([abs(n - target) for n in counts]))
                chosen[alpha] = cands[best]
                spread.append(counts[best])
            entry['density_target'] = target
            entry['density_achieved'] = spread
            if max(spread) - min(spread) > args.match_tol:
                print(
                    f'  {stem}: density spread {max(spread) - min(spread)} notes exceeds tol {args.match_tol}, flagged in manifest'
                )
                entry['density_matched'] = False
            else:
                entry['density_matched'] = True
        else:
            chosen = {alpha: sample(alpha, base_rng) for alpha in args.alphas}
        for alpha in args.alphas:
            out_tokens = chosen[alpha]
            label = f'a{alpha:+.1f}'.replace('.', 'p').replace('+', 'p').replace('-', 'm')
            rel = f'{stem}/{label}.mid'
            tokens_to_midi(out_tokens, grid_hz=grid_hz).write(str(args.out / rel))
            entry['clips'].append(
                {
                    'alpha': alpha,
                    'file': rel,
                    'descriptors': {k: round(float(v), 4) for k, v in describe(out_tokens).items()},
                }
            )
        manifest['seeds'].append(entry)
        print(f'{stem}  side={side:<9} index={index:<7} {source[:60]}')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(
        f'\nwrote {len(picks)} seeds x {len(args.alphas)} clips to {args.out} ({args.bars} bars = {clip_steps / grid_hz:.1f}s each)'
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path


def note_count(clip):
    return round(clip['descriptors']['note_density'] * 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', type=Path, default=Path('artifacts/demos_raw_v13'))
    ap.add_argument('--audio', type=Path, default=None, help='default: <demos>_audio')
    ap.add_argument('--out', type=Path, default=Path('listening/ab_test'))
    ap.add_argument('--alpha', type=float, default=3.0, help='compare -alpha against +alpha')
    ap.add_argument('--tol', type=int, default=2, help='largest note-count difference allowed inside a pair')
    ap.add_argument('--pairs', type=int, default=24)
    ap.add_argument('--max-reuse', type=int, default=2, help='times one seed may appear on the same side')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--same-seed',
        action='store_true',
        help="pair each melody's -alpha clip with its own +alpha clip instead of matching different melodies by note count",
    )
    ap.add_argument(
        '--balance-density',
        action='store_true',
        help='with --same-seed: drop pairs until the jazzier clip is denser as often as it is sparser',
    )
    args = ap.parse_args()
    audio = args.audio or args.demos.with_name(args.demos.name + '_audio')
    manifest = json.loads((args.demos / 'manifest.json').read_text())
    neg, pos = ({}, {})
    for s in manifest['seeds']:
        for c in s['clips']:
            if c['alpha'] == -args.alpha:
                neg[s['id']] = c
            elif c['alpha'] == args.alpha:
                pos[s['id']] = c
    if args.same_seed:
        cands = [
            (s, s, note_count(neg[s]), note_count(pos[s]))
            for s in neg
            if s in pos and abs(note_count(neg[s]) - note_count(pos[s])) <= args.tol
        ]
        if args.balance_density:
            up = [c for c in cands if c[3] > c[2]]
            down = [c for c in cands if c[3] < c[2]]
            tied = [c for c in cands if c[3] == c[2]]
            k = min(len(up), len(down))
            cands = tied + up[:k] + down[:k]
        args.pairs = min(args.pairs, len(cands))
    else:
        cands = [
            (i, j, note_count(neg[i]), note_count(pos[j]))
            for i in neg
            for j in pos
            if i != j and abs(note_count(neg[i]) - note_count(pos[j])) <= args.tol
        ]
    if not cands:
        raise SystemExit(f'no pairs within {args.tol} notes at alpha {args.alpha}')
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    cands.sort(key=lambda p: abs(p[2] - p[3]))
    chosen, used_neg, used_pos = ([], {}, {})
    for i, j, a, b in cands:
        if used_neg.get(i, 0) >= args.max_reuse or used_pos.get(j, 0) >= args.max_reuse:
            continue
        chosen.append((i, j, a, b))
        used_neg[i] = used_neg.get(i, 0) + 1
        used_pos[j] = used_pos.get(j, 0) + 1
        if len(chosen) == args.pairs:
            break
    if args.out.exists():
        raise SystemExit(f'{args.out} exists, move it aside first')
    args.out.mkdir(parents=True)
    key = []
    for n, (i, j, a, b) in enumerate(chosen, 1):
        wav_neg = audio / Path(neg[i]['file']).with_suffix('.wav')
        wav_pos = audio / Path(pos[j]['file']).with_suffix('.wav')
        jazzy_is_a = rng.random() < 0.5
        shutil.copy(wav_pos if jazzy_is_a else wav_neg, args.out / f'pair{n:02d}_A.wav')
        shutil.copy(wav_neg if jazzy_is_a else wav_pos, args.out / f'pair{n:02d}_B.wav')
        key.append(
            {
                'pair': n,
                'jazzier': 'A' if jazzy_is_a else 'B',
                'notes_A': b if jazzy_is_a else a,
                'notes_B': a if jazzy_is_a else b,
                'neg_seed': i,
                'pos_seed': j,
            }
        )
    (args.out / 'ANSWER_KEY.json').write_text(
        json.dumps({'source': str(args.demos), 'alpha': args.alpha, 'tol': args.tol, 'pairs': key}, indent=2)
    )
    diffs = [abs(k['notes_A'] - k['notes_B']) for k in key]
    ties = sum((d == 0 for d in diffs))
    jazzy_denser = sum(
        (k['notes_A'] > k['notes_B'] if k['jazzier'] == 'A' else k['notes_B'] > k['notes_A'] for k in key)
    )
    print(f'wrote {len(key)} pairs to {args.out}')
    print(
        f'note-count difference within a pair: mean {sum(diffs) / len(diffs):.1f}, max {max(diffs)}, exact ties {ties}/{len(key)}'
    )
    print(
        f'jazzier clip is the denser one in {jazzy_denser}/{len(key)} pairs ({len(key) - ties - jazzy_denser} the other way, {ties} tied), so counting notes is no longer a winning strategy'
    )
    print(f"jazzy side is A in {sum((k['jazzier'] == 'A' for k in key))}/{len(key)}")
    print(f'chance is 50%. Score yourself on {len(key)} pairs, then read {args.out}/ANSWER_KEY.json')


if __name__ == '__main__':
    main()

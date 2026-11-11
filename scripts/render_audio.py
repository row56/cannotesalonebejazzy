#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path
import numpy as np

MACOS_GM = Path('/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls')
BUNDLED_SYNTH = sorted(
    Path('/opt/homebrew/Cellar/fluid-synth').glob('*/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2')
)


def find_bank() -> Path:
    user = Path.home() / 'soundfonts'
    hits = sorted(user.glob('*.sf2')) + sorted(user.glob('*.sf3')) if user.is_dir() else []
    if hits:
        return hits[0]
    if MACOS_GM.exists():
        return MACOS_GM
    if BUNDLED_SYNTH:
        print(
            'warning: falling back to Vintage Dreams Waves, a synth bank. Its preset 0 is not a piano and the clips will sound wrong.',
            file=sys.stderr,
        )
        return BUNDLED_SYNTH[-1]
    raise SystemExit('no soundfont found; pass --soundfont')


def read_wav(path: Path):
    with wave.open(str(path)) as w:
        params = w.getparams()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (params, data.astype(np.float32))


def write_wav(path: Path, params, data: np.ndarray) -> None:
    with wave.open(str(path), 'wb') as w:
        w.setparams(params)
        w.writeframes(np.clip(data, -32768, 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path, help='directory of .mid files, searched recursively')
    ap.add_argument('--out', type=Path, default=None, help='default: <src>_audio')
    ap.add_argument('--soundfont', type=Path, default=None)
    ap.add_argument('--peak-db', type=float, default=-3.0, help='target peak for the loudest clip in the set')
    ap.add_argument('--rate', type=int, default=44100)
    ap.add_argument(
        '--program',
        type=int,
        default=None,
        metavar='GM',
        help='override the GM program before synthesis, e.g. 56 for trumpet. Piano decays, so a held note stops sounding well before it ends and the HOLD tokens are lost; a sustaining monophonic instrument keeps them audible',
    )
    ap.add_argument(
        '--count-in', type=int, default=0, metavar='BEATS', help='metronome clicks before the melody starts. 0 disables'
    )
    ap.add_argument(
        '--bpm', type=float, default=None, help='tempo of the count-in; default: bpm from <src>/manifest.json'
    )
    args = ap.parse_args()
    if args.count_in and args.bpm is None:
        manifest = args.src / 'manifest.json'
        if not manifest.exists():
            raise SystemExit('--count-in needs --bpm or a manifest.json with bpm')
        import json

        args.bpm = float(json.loads(manifest.read_text())['bpm'])
    if shutil.which('fluidsynth') is None:
        raise SystemExit('fluidsynth not on PATH (brew install fluid-synth)')
    bank = args.soundfont or find_bank()
    out_root = args.out or args.src.with_name(args.src.name + '_audio')
    midis = sorted(args.src.rglob('*.mid'))
    if not midis:
        raise SystemExit(f'no .mid files under {args.src}')
    print(f'{len(midis)} files, bank {bank.name}')
    tmp_dir = None
    if args.program is not None or args.count_in:
        import tempfile
        import pretty_midi

        tmp_dir = Path(tempfile.mkdtemp(prefix='mvae_render_'))
        if args.program is not None:
            print(f'overriding GM program -> {args.program}')
        if args.count_in:
            print(f'count-in: {args.count_in} beats at {args.bpm:g} bpm')
    written = []
    for n, mid in enumerate(midis, 1):
        dest = out_root / mid.relative_to(args.src).with_suffix('.wav')
        dest.parent.mkdir(parents=True, exist_ok=True)
        src_mid = mid
        if tmp_dir is not None:
            pm = pretty_midi.PrettyMIDI(str(mid))
            if args.program is not None:
                for inst in pm.instruments:
                    inst.program = args.program
                    inst.is_drum = False
            if args.count_in:
                beat = 60.0 / args.bpm
                shift = args.count_in * beat
                for inst in pm.instruments:
                    for note in inst.notes:
                        note.start += shift
                        note.end += shift
                click = pretty_midi.Instrument(program=0, is_drum=True, name='count-in')
                for b in range(args.count_in):
                    click.notes.append(
                        pretty_midi.Note(
                            velocity=100 if b == 0 else 80,
                            pitch=76 if b == 0 else 77,
                            start=b * beat,
                            end=b * beat + 0.1,
                        )
                    )
                pm.instruments.append(click)
            src_mid = tmp_dir / f'{n:05d}.mid'
            pm.write(str(src_mid))
        proc = subprocess.run(
            ['fluidsynth', '-ni', '-F', str(dest), '-r', str(args.rate), '-g', '1.0', str(bank), str(src_mid)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not dest.exists():
            sys.exit(f'fluidsynth failed on {mid}:\n{proc.stderr}')
        written.append(dest)
        if n % 25 == 0 or n == len(midis):
            print(f'  rendered {n}/{len(midis)}')
    peak = max((float(np.abs(read_wav(p)[1]).max()) for p in written))
    if peak <= 0:
        raise SystemExit('every rendered file is silent; wrong bank or empty MIDI')
    target = 32767 * 10 ** (args.peak_db / 20)
    gain = target / peak
    print(f'set peak {peak:.0f} -> applying shared gain {gain:.2f}')
    for p in written:
        params, data = read_wav(p)
        write_wav(p, params, data * gain)
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'wrote {len(written)} wav files to {out_root}')


if __name__ == '__main__':
    main()

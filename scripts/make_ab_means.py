import json
import random
import subprocess
import sys
from pathlib import Path
import pretty_midi

src, out, sf2 = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
cross = '--cross' in sys.argv
m = json.load(open(src / 'manifest.json'))
bpm = m['bpm']
beat = 60.0 / bpm
out.mkdir(parents=True, exist_ok=True)
(out / 'answer_key_DONT_OPEN').mkdir(exist_ok=True)
rng = random.Random(2027)
seeds = m['seeds'][:]
rng.shuffle(seeds)


def render(mid_path, wav_path):
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    song = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0)
    for n in pm.instruments[0].notes:
        piano.notes.append(pretty_midi.Note(78, n.pitch, n.start + 4 * beat, n.end + 4 * beat))
    click = pretty_midi.Instrument(program=0, is_drum=True)
    for b in range(4):
        click.notes.append(pretty_midi.Note(70 if b == 0 else 50, 37, b * beat, b * beat + 0.08))
    song.instruments += [piano, click]
    tmp = wav_path.with_suffix('.mid')
    song.write(str(tmp))
    subprocess.run(
        [
            'fluidsynth',
            '-ni',
            '-g',
            '0.5',
            '-R',
            '1',
            '-C',
            '0',
            '-o',
            'synth.reverb.room-size=0.45',
            '-o',
            'synth.reverb.level=0.35',
            '-F',
            str(wav_path),
            '-r',
            '44100',
            sf2,
            str(tmp),
        ],
        check=True,
        capture_output=True,
    )
    tmp.unlink()


key = []
for k, s in enumerate(seeds, 1):
    clips = s['clips']
    if cross:
        steered = next(iter(clips))
        pair = {'original': f"{s['id']}/original.mid", 'steered': clips[steered]['file']}
        jazzier = 'steered' if s['side'] == 'classical' else 'original'
    else:
        pair = {'classical': clips['classical']['file'], 'jazz': clips['jazz']['file']}
        jazzier = 'jazz'
    names = list(pair)
    rng.shuffle(names)
    for lab, which in zip('AB', names):
        render(src / pair[which], out / f'pair{k:02d}_{lab}.wav')
    key.append(
        {
            'pair': k,
            'jazzier': 'AB'[names.index(jazzier)],
            'A': names[0],
            'B': names[1],
            'seed': s['id'],
            'seed_side': s['side'],
            'source_file': s['source_file'],
            'start_sd': round(s['start_sd'], 2),
            'notes': s['notes'],
        }
    )
(out / 'answer_key_DONT_OPEN' / 'key.json').write_text(json.dumps(key, indent=2))
(out / 'answers.txt').write_text(
    'which clip is jazzier? write A or B after each pair\n\n'
    + ''.join((f'pair{k:02d}: \n' for k in range(1, len(seeds) + 1)))
)
print(len(seeds), 'pairs')

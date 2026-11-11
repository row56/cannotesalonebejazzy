#!/usr/bin/env python3
import argparse
import csv
import hashlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.labels import UNKNOWN, era_of, normalise


@dataclass
class Source:
    genre: str
    artist_from: str
    where: str
    url: str = None


SOURCES = {
    'giantmidi': Source(
        genre='classical',
        artist_from='filename_comma',
        where="GiantMIDI-Piano (ByteDance), transcribed with Kong et al.'s\nhigh-resolution piano transcription - the same system behind\nPiJAMA's midi_kong archive.\n    https://github.com/bytedance/GiantMIDI-Piano\nNo direct link on purpose: the 193 MB archive sits behind a\nliability disclaimer in disclaimer.md that YOU have to accept, on\na Google Drive folder (so fetch it by hand and pass --archive).\nThe repo also ships a script that rebuilds the set from YouTube\naudio if the mirror is down - about 10 days on one GPU.\nFilenames carry the composer, e.g. 'Mozart, Wolfgang Amadeus, Sonata..., <id>.mid' - artist is the text before the first comma.",
    ),
    'pijama': Source(
        genre='jazz',
        artist_from='performer_album',
        where='PiJAMA: Piano Jazz with Automatic MIDI Annotations (Edwards, Dixon, Benetos, ISMIR 2023). Zenodo record 8354955:\n    https://zenodo.org/records/8354955\nCC-BY-NC-4.0. Two archives, same audio, DIFFERENT transcribers:\n  midi_kong.zip       Kong et al. high-resolution piano transcription\n  midi_hawthorne.zip  Onsets-and-Frames\nUse midi_kong: GiantMIDI was transcribed with the same system, so\nmatching it is what stops the genre axis from being a transcriber\naxis. See the confound note in the README.\nLayout is <live|studio>/<Performer>/<Album>/<track>.midi.',
        url='https://zenodo.org/api/records/8354955/files/midi_kong.zip/content',
    ),
    'maestro': Source(
        genre='classical',
        artist_from='maestro_csv',
        where='MAESTRO v3.0.0 (Hawthorne et al., ICLR 2019), MIDI only, 58 MB:\n    https://magenta.tensorflow.org/datasets/maestro\nCC-BY-NC-SA-4.0. This is the confound control: unlike GiantMIDI and\nPiJAMA it is NOT transcribed from audio, it is Disklavier capture,\nso the MIDI is ground truth. Classical from a second, independent\npipeline is what breaks the genre/corpus collinearity - see the\nconfound note in the README.\nLayout is <year>/<track>.midi with composers in maestro-v3.0.0.csv.',
        url='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
    ),
}
_MAESTRO_INDEX: dict[Path, dict[str, str]] = {}


def maestro_index(staging: Path):
    if staging not in _MAESTRO_INDEX:
        csvs = sorted(staging.rglob('maestro-v*.csv'))
        if not csvs:
            raise SystemExit(f'no maestro-v*.csv under {staging}; wrong archive?')
        with csvs[0].open(newline='') as fh:
            _MAESTRO_INDEX[staging] = {
                r['midi_filename']: r['canonical_composer'] or UNKNOWN for r in csv.DictReader(fh)
            }
        print(f'read {len(_MAESTRO_INDEX[staging])} composer rows from {csvs[0].name}')
    return _MAESTRO_INDEX[staging]


def show_sources():
    for key, src in SOURCES.items():
        print(f'\n=== {key}  (genre: {src.genre}) ===')
        print(src.where)
        print(f"url currently configured: {src.url or '<none - pass --url>'}")
    print()


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={'User-Agent': 'style-axes/0.1'})
    if have:
        req.add_header('Range', f'bytes={have}-')
        print(f'resuming at {have / 1000000.0:.1f} MB')
    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:
            print('server ignored the range request, restarting')
            have = 0
        total = int(resp.headers.get('Content-Length', 0)) + have
        mode = 'ab' if have and resp.status == 206 else 'wb'
        done = have if mode == 'ab' else 0
        with part.open(mode) as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                pct = f'{100 * done / total:5.1f}%' if total else '  ?  '
                print(f'\r  {done / 1000000.0:8.1f} MB  {pct}', end='', flush=True)
    print()
    part.rename(dest)
    return dest


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f'extracting {archive.name} -> {dest}')
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if _unsafe(info.filename) or info.is_dir():
                    continue
                target = dest / _zip_name(info)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open('wb') as fh:
                    shutil.copyfileobj(src, fh)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers() if not _unsafe(m.name)]
            tf.extractall(dest, members=members)
    else:
        raise SystemExit(f'{archive} is neither zip nor tar')


def _unsafe(name: str):
    p = Path(name)
    return p.is_absolute() or '..' in p.parts


def _zip_name(info: zipfile.ZipInfo):
    if info.flag_bits & 2048:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def _slug(name: str):
    return normalise(name).replace(' ', '_') or UNKNOWN


def fields_of(path: Path, root: Path, src: Source):
    rel = path.relative_to(root)
    parts = rel.parts
    if src.artist_from == 'filename_comma':
        return {'artist': path.stem.split(',')[0].strip() or UNKNOWN, 'album': UNKNOWN, 'setting': UNKNOWN}
    if src.artist_from == 'performer_album':
        return {
            'artist': parts[-3] if len(parts) >= 3 else UNKNOWN,
            'album': parts[-2] if len(parts) >= 2 else UNKNOWN,
            'setting': parts[-4] if len(parts) >= 4 else UNKNOWN,
        }
    if src.artist_from == 'maestro_csv':
        key = '/'.join(parts[-2:])
        return {'artist': maestro_index(root).get(key, UNKNOWN), 'album': UNKNOWN, 'setting': UNKNOWN}
    return {'artist': UNKNOWN, 'album': UNKNOWN, 'setting': UNKNOWN}


MANIFEST = '_manifest.csv'
MANIFEST_FIELDS = ['path', 'artist', 'album', 'setting']


def organise(staging: Path, raw: Path, key: str, src: Source):
    midis = sorted((p for p in staging.rglob('*') if p.suffix.lower() in {'.mid', '.midi'}))
    if not midis:
        raise SystemExit(f'no midi files found under {staging}')
    out = raw / src.genre / key
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows, collisions = ([], 0)
    for path in midis:
        f = fields_of(path, staging, src)
        target_dir = out / _slug(f['artist']) / _slug(f['album'])
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        if target.exists():
            stem, suffix = (target.stem, target.suffix)
            n = 2
            while target.exists():
                target = target_dir / f'{stem}__{n}{suffix}'
                n += 1
            collisions += 1
        shutil.copy2(path, target)
        rows.append({'path': str(target.relative_to(raw)), **f})
    with (out / MANIFEST).open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'organised {len(rows)} files into {out}')
    if collisions:
        print(f'  {collisions} duplicate track titles kept under __N suffixes')
    return len(rows)


def write_metadata(raw: Path):
    rows = []
    for genre_dir in sorted((p for p in raw.iterdir() if p.is_dir())):
        for corpus_dir in sorted((p for p in genre_dir.iterdir() if p.is_dir())):
            manifest = corpus_dir / MANIFEST
            if not manifest.exists():
                print(f'  skipping {corpus_dir}: no {MANIFEST}, re-run its ingest')
                continue
            with manifest.open(newline='') as fh:
                for src_row in csv.DictReader(fh):
                    artist = src_row['artist']
                    rows.append(
                        {
                            'path': src_row['path'],
                            'corpus': corpus_dir.name,
                            'genre': genre_dir.name,
                            'artist': artist,
                            'album': src_row['album'],
                            'setting': src_row['setting'],
                            'era': era_of(artist) if genre_dir.name == 'classical' else UNKNOWN,
                        }
                    )
    if not rows:
        raise SystemExit(f'no midi files under {raw}; nothing to describe')
    dest = raw / 'metadata.csv'
    with dest.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwrote {dest}  ({len(rows)} files)')
    for field in ('genre', 'corpus', 'era', 'setting'):
        tally = {}
        for r in rows:
            tally[r[field]] = tally.get(r[field], 0) + 1
        print(f'  {field}: ' + ', '.join((f'{k}={v}' for k, v in sorted(tally.items()))))
    artists = {r['artist'] for r in rows}
    print(f'  artists: {len(artists)} distinct')
    unknown = sum((1 for r in rows if r['era'] == UNKNOWN and r['genre'] == 'classical'))
    if unknown:
        print(f'  {unknown} classical files have no era; extend COMPOSER_ERA in labels.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(SOURCES))
    ap.add_argument('--url', help='direct download URL (overrides SOURCES)')
    ap.add_argument('--archive', type=Path, help='use an already-downloaded archive')
    ap.add_argument('--sha256', help='verify the archive against this digest')
    ap.add_argument('--raw', type=Path, default=Path('data/raw'))
    ap.add_argument('--cache', type=Path, default=Path('data/archives'))
    ap.add_argument('--show', action='store_true', help='print where to get each corpus')
    ap.add_argument(
        '--metadata-only', action='store_true', help='rebuild metadata.csv from whatever is already in data/raw'
    )
    args = ap.parse_args()
    if args.show:
        show_sources()
        return
    if args.metadata_only:
        write_metadata(args.raw)
        return
    if not args.dataset:
        ap.error('--dataset is required (or use --show / --metadata-only)')
    src = SOURCES[args.dataset]
    url = args.url or src.url
    if args.archive:
        archive = args.archive
        if not archive.exists():
            raise SystemExit(f'{archive} does not exist')
    elif url:
        archive = args.cache / f'{args.dataset}{_suffix(url)}'
        if archive.exists():
            print(f'using cached {archive}')
        else:
            download(url, archive)
    else:
        print(src.where)
        raise SystemExit(
            f'\nno URL configured for {args.dataset}. Pass --url once you have it, or --archive if you downloaded it by hand.'
        )
    if args.sha256:
        digest = sha256(archive)
        if digest != args.sha256:
            raise SystemExit(f'checksum mismatch\n  expected {args.sha256}\n  got      {digest}')
        print('checksum ok')
    staging = args.cache / f'{args.dataset}_staging'
    if staging.exists():
        shutil.rmtree(staging)
    extract(archive, staging)
    organise(staging, args.raw, args.dataset, src)
    shutil.rmtree(staging, ignore_errors=True)
    write_metadata(args.raw)


def _suffix(url: str):
    for ext in ('.zip', '.tar.gz', '.tgz', '.tar'):
        if url.lower().split('?')[0].endswith(ext):
            return ext
    return '.zip'


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import csv
import re
import zipfile
from pathlib import Path

FIELDS = ['path', 'corpus', 'genre', 'artist', 'album', 'setting', 'era']
UNKNOWN = 'unknown'


def split_camel(name: str) -> str:
    spaced = re.sub('(?<=[a-z])(?=[A-Z])', ' ', name)
    return re.sub('(?<=[A-Z])(?=[A-Z][a-z])', ' ', spaced).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', type=Path, default=Path('data/archives/wjazzd_mid_unquant.zip'))
    ap.add_argument('--raw', type=Path, default=Path('data/raw'))
    ap.add_argument('--corpus', default='wjazzd')
    args = ap.parse_args()
    dest_root = args.raw / 'jazz' / args.corpus
    rows = []
    with zipfile.ZipFile(args.zip) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(('.mid', '.midi')):
                continue
            stem = Path(info.filename).name
            artist = split_camel(stem.split('_')[0])
            out = dest_root / artist.replace(' ', '_') / stem
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(info))
            rows.append(
                {
                    'path': str(out.relative_to(args.raw)),
                    'corpus': args.corpus,
                    'genre': 'jazz',
                    'artist': artist,
                    'album': UNKNOWN,
                    'setting': UNKNOWN,
                    'era': UNKNOWN,
                }
            )
    rows.sort(key=lambda r: r['path'])
    meta = args.raw / 'metadata.csv'
    existing = list(csv.DictReader(meta.open(newline=''))) if meta.exists() else []
    kept = [r for r in existing if r.get('corpus') != args.corpus]
    with meta.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept + rows)
    solo = args.raw.parent / 'raw_wjazzd'
    solo.mkdir(parents=True, exist_ok=True)
    link = solo / 'jazz'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((args.raw / 'jazz').resolve())
    with (solo / 'metadata.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    artists = sorted({r['artist'] for r in rows})
    print(f'wrote {len(rows)} midi files under {dest_root}')
    print(f'metadata.csv: {len(kept)} kept + {len(rows)} {args.corpus}')
    print(f'{len(artists)} artists: ' + ', '.join(artists[:6]) + ', ...')
    print(f'standalone preprocess root: {solo}')


if __name__ == '__main__':
    main()

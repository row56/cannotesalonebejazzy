#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.train import run


def coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return {'true': True, 'false': False}.get(value.lower(), value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('configs/base.yaml'))
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for pair in args.set:
        key, _, value = pair.partition('=')
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = coerce(value)
    print(yaml.safe_dump(cfg, sort_keys=False))
    ckpt = run(cfg)
    print(f'best checkpoint: {ckpt}')


if __name__ == '__main__':
    main()

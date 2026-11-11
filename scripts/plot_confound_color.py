import argparse
import sys
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
INK, MUTED, GRID = ('#0b0b0b', '#52514e', '#c9c9c5')
CLASSICAL = {'facecolor': '#24479A', 'edge': '#24479A', 'hatch': None, 'linestyle': 'solid'}
JAZZ = {'facecolor': '#B4501A', 'edge': '#B4501A', 'hatch': None, 'linestyle': 'solid'}
CORPORA = [
    ('lieder', 'Lieder', 'classical', 'notated'),
    ('quartets', 'quartets', 'classical', 'notated'),
    ('asap', 'ASAP', 'classical', 'transcribed'),
    ('jtd', 'JTD', 'jazz', 'transcribed'),
    ('wjazzd', 'WJazzD', 'jazz', 'transcribed'),
]
POLE_LO, POLE_HI = ('lieder', 'wjazzd')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v5', default='runs/base_v5/axes_heldout/projections.npz')
    ap.add_argument('--out', default='paper/fig_confound.pdf')
    args = ap.parse_args()
    d = np.load(args.v5, allow_pickle=True)
    names = [str(n) for n in d['corpus_names']]
    codes = d['corpus']
    per = {nm: d['projection'][codes == i] for i, nm in enumerate(names) if (codes == i).any()}
    lo, hi = (per[POLE_LO].mean(), per[POLE_HI].mean())
    per = {k: (v - lo) / (hi - lo) for k, v in per.items()}
    plt.rcParams.update(
        {
            'font.size': 6.4,
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'hatch.linewidth': 0.4,
        }
    )
    fig, ax = plt.subplots(figsize=(3.4, 1.85), constrained_layout=True)
    grid = np.linspace(-1.7, 2.6, 420)
    gap = 0.45
    positions, labels = ([], [])
    y = 0.0
    prev_fmt = None
    for name, label, genre, fmt in reversed(CORPORA):
        if prev_fmt is not None and fmt != prev_fmt:
            y += gap
        prev_fmt = fmt
        style = CLASSICAL if genre == 'classical' else JAZZ
        dens = gaussian_kde(per[name])(grid)
        dens = dens / dens.max() * 0.82
        ax.fill_between(grid, y, y + dens, facecolor=style['facecolor'], alpha=0.35, lw=0, zorder=2)
        ax.plot(grid, y + dens, color=style['edge'], lw=1.1, linestyle=style['linestyle'], zorder=3)
        positions.append(y)
        labels.append(label)
        y += 1.0
    tr = [p for p, (_, _, _, f) in zip(positions, reversed(CORPORA)) if f == 'transcribed']
    no = [p for p, (_, _, _, f) in zip(positions, reversed(CORPORA)) if f == 'notated']
    for rows, text in ((tr, 'transcribed'), (no, 'notated')):
        ax.text(
            2.72,
            (min(rows) + max(rows)) / 2 + 0.35,
            text,
            rotation=270,
            va='center',
            ha='left',
            fontsize=6.0,
            color=MUTED,
        )
        ax.plot([2.64, 2.64], [min(rows) - 0.06, max(rows) + 0.86], color=GRID, lw=0.8, clip_on=False)
    ax.set_yticks([p + 0.06 for p in positions])
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(-1.7, 2.6)
    ax.set_ylim(-0.15, y + 0.12)
    ax.set_xticks([])
    ax.tick_params(length=0, colors=MUTED)
    ax.set_xlabel("projections onto the first model's jazz axis", color=MUTED, fontsize=6.2)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    handles = [
        Patch(facecolor=s['facecolor'], alpha=0.5, edgecolor=s['edge'], lw=0.9, label=lab)
        for s, lab in ((CLASSICAL, 'classical'), (JAZZ, 'jazz'))
    ]
    ax.legend(
        handles=handles,
        loc='upper left',
        frameon=False,
        fontsize=6.0,
        handlelength=2.2,
        handleheight=1.1,
        labelcolor=MUTED,
        borderaxespad=0.1,
        labelspacing=0.45,
    )
    fig.savefig(args.out, bbox_inches='tight')
    png = str(Path(args.out).with_suffix('.png'))
    fig.savefig(png, dpi=900, bbox_inches='tight')
    print(f'wrote {args.out} and {png}')


if __name__ == '__main__':
    main()

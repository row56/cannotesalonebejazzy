import argparse
import sys
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CLASSICAL, JAZZ = ('#2a78d6', '#eb6834')
INK, MUTED, GRID = ('#0b0b0b', '#52514e', '#d8d8d4')
ROWS = [
    ('Lieder', 'lieder', 'lieder', 'classical'),
    ('ASAP', 'asap', 'asap_test', 'classical'),
    ('quartets', 'quartets', 'quartets', 'classical'),
    ('PDMX jazz', None, 'pdmx_jazz_test', 'jazz'),
    ('JTD', 'jtd', 'jtd_test', 'jazz'),
    ('WJazzD', 'wjazzd', 'wjazzd', 'jazz'),
]
FIT = {'v5': {'lieder', 'wjazzd'}, 'v6': {'lieder', 'asap', 'pdmx_jazz', 'jtd'}}
POLES = {'v5': (['lieder'], ['wjazzd']), 'v6': (['lieder', 'asap'], ['pdmx_jazz', 'jtd'])}


def load(path):
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d['corpus_names']]
    codes = d['corpus']
    return {nm: d['projection'][codes == i] for i, nm in enumerate(names) if (codes == i).any()}


def rescale(panel, per_corpus):
    cl, jz = POLES[panel]
    lo = np.concatenate([per_corpus[c] for c in cl]).mean()
    hi = np.concatenate([per_corpus[c] for c in jz]).mean()
    return {k: (v - lo) / (hi - lo) for k, v in per_corpus.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v5', default='runs/base_v5/axes_heldout/projections.npz')
    ap.add_argument('--v6', default='runs/base_v6/axes_dm_heldout/projections.npz')
    ap.add_argument('--out', default='paper/fig_projections.pdf')
    args = ap.parse_args()
    data = {'v6': rescale('v6', load(args.v6))}
    plt.rcParams.update({'font.size': 6.4, 'font.family': 'sans-serif', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.0), constrained_layout=True)
    axes = [ax]
    grid = np.linspace(-1.7, 2.7, 420)
    for ax, panel in zip(axes, ['v6']):
        ax.axvline(0, color=GRID, lw=0.7, zorder=0)
        ax.axvline(1, color=GRID, lw=0.7, zorder=0)
        for row, (label, c5, c6, genre) in enumerate(ROWS):
            name = c5 if panel == 'v5' else c6
            if name is None or name not in data[panel]:
                continue
            v = data[panel][name]
            colour = CLASSICAL if genre == 'classical' else JAZZ
            held = name not in FIT[panel]
            dens = gaussian_kde(v)(grid)
            dens = dens / dens.max() * 0.78
            base = len(ROWS) - 1 - row
            ax.fill_between(grid, base, base + dens, color=colour, alpha=0.55 if held else 0.16, lw=0, zorder=2)
            ax.plot(grid, base + dens, color=colour, lw=0.9, ls='-' if held else (0, (2.2, 1.4)), zorder=3)
        ax.set_xlim(-1.6, 2.6)
        ax.set_ylim(-0.2, len(ROWS) - 0.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['classical\ncentroid', 'jazz\ncentroid'], color=MUTED)
        ax.tick_params(length=0, colors=MUTED)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
    axes[0].set_yticks([len(ROWS) - 1 - i for i in range(len(ROWS))])
    axes[0].set_yticklabels([r[0] for r in ROWS], color=INK)
    handles = [
        plt.Line2D([], [], color=CLASSICAL, lw=3, alpha=0.55, label='classical'),
        plt.Line2D([], [], color=JAZZ, lw=3, alpha=0.55, label='jazz'),
        plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (2.2, 1.4)), label='in the axis fit'),
        plt.Line2D([], [], color=MUTED, lw=0.9, label='held out'),
    ]
    fig.legend(
        handles=handles,
        loc='outside lower center',
        ncol=4,
        frameon=False,
        fontsize=5.9,
        handlelength=1.5,
        columnspacing=1.1,
        labelcolor=MUTED,
    )
    fig.savefig(args.out, bbox_inches='tight')
    png = str(Path(args.out).with_suffix('.png'))
    fig.savefig(png, dpi=320, bbox_inches='tight')
    print(f'wrote {args.out} and {png}')


if __name__ == '__main__':
    main()

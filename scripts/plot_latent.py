import argparse
import sys
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melodicvae.data import MelodyDataset
from melodicvae.model import MelodyVAE, VAEConfig
from melodicvae.style import StyleSpace, encode_dataset
from melodicvae.train import pick_device

CLASSICAL, JAZZ = ('#2a78d6', '#eb6834')
INK, MUTED, GRID = ('#0b0b0b', '#52514e', '#d8d8d4')
AB_ROUND2 = ['data/processed/train_v6.npz:6835', 'data/processed/train_v7.npz:1906', 'data/processed/train_v7.npz:2021']


def pca_basis(zc: np.ndarray, k: int = 2):
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    return vt[:k].T


def axis_basis(zc: np.ndarray, axis: np.ndarray):
    resid = zc - np.outer(zc @ axis, axis)
    return np.column_stack([axis, pca_basis(resid, 1)[:, 0]])


def subsample(idx: np.ndarray, n: int, rng):
    return rng.choice(idx, size=n, replace=False) if len(idx) > n else idx


def pick_seeds(z: np.ndarray, mask: np.ndarray, n: int):
    pool = np.flatnonzero(mask)
    d = np.linalg.norm(z[pool] - z[pool].mean(0), axis=1)
    return pool[np.argsort(d)[:n]]


@torch.no_grad()
def encode_windows(model, specs: list[str], device: str):
    out, tags = ([], [])
    for spec in specs:
        path, _, i = spec.rpartition(':')
        i = int(i)
        blob = np.load(path, allow_pickle=True)
        w = torch.tensor(blob['x'][i].astype(np.int64), device=device).unsqueeze(0)
        mu, _ = model.encode(w)
        out.append(mu.cpu().numpy()[0])
        src = str(blob['source_file'][i]) if 'source_file' in blob.files else ''
        tags.append(f'{Path(path).stem}[{i}] {src}')
    return (np.stack(out), tags)


def draw(ax, p, gz, cents, seeds_2d, alphas, xlabel, ylabel, title):
    ax.scatter(*p[gz == 0].T, s=5, alpha=0.35, c=CLASSICAL, lw=0, zorder=1)
    ax.scatter(*p[gz == 1].T, s=5, alpha=0.35, c=JAZZ, lw=0, zorder=1)
    for c, colour in zip(cents, (CLASSICAL, JAZZ)):
        ax.scatter(*c, s=90, marker='X', c=colour, edgecolors='white', linewidths=1.0, zorder=7)
    for traj in seeds_2d:
        ax.plot(*traj.T, color='white', lw=2.6, zorder=3, solid_capstyle='round')
        ax.plot(*traj.T, color=INK, lw=1.1, zorder=4, solid_capstyle='round')
        for a, mark in ((alphas[0], 'v'), (0.0, 'o'), (alphas[-1], '^')):
            j = int(np.argmin(np.abs(alphas - a)))
            ax.scatter(*traj[j], s=34, marker=mark, c='white', edgecolors=INK, linewidths=0.9, zorder=6)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, fontsize=7.2, pad=4)
    ax.tick_params(length=0, colors=MUTED, labelsize=5.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/base_v6/best.pt')
    ap.add_argument('--data', default='data/processed/eval_v7.npz')
    ap.add_argument('--styles', default='runs/base_v6/axes_dm_heldout/styles.npz')
    ap.add_argument('--axis', default='jazz')
    ap.add_argument(
        '--n-per-genre', type=int, default=1500, help='windows scattered per genre; all of them are encoded regardless'
    )
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument(
        '--seeds', type=int, nargs='*', default=None, help='explicit window indices into --data; overrides --n-seeds'
    )
    ap.add_argument(
        '--seed-windows',
        nargs='*',
        default=AB_ROUND2,
        help='seeds as path.npz:index, for melodies that live in another file; defaults to the first three A/B round-2 pairs',
    )
    ap.add_argument('--alpha-max', type=float, default=3.0)
    ap.add_argument('--n-alphas', type=int, default=13)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--panel',
        choices=('both', 'pca', 'axis'),
        default='both',
        help="'axis' is the standalone talk figure; 'both' is the paper figure",
    )
    ap.add_argument('--out', default='paper/fig_latent.pdf')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    blob = torch.load(args.ckpt, map_location=device)
    model = MelodyVAE(VAEConfig(**blob['cfg'].get('model', {})))
    model.load_state_dict(blob['model'])
    ds = MelodyDataset(args.data, augment_transpose=0)
    z = encode_dataset(model, DataLoader(ds, batch_size=256), device)
    print(f'encoded {len(z)} windows into {z.shape[1]}-d latents')
    space = StyleSpace.load(args.styles)
    if args.axis not in space.axes:
        raise SystemExit(f'unknown axis {args.axis!r}; have {list(space.axes)}')
    axis = space.axes[args.axis]
    sigma = float((z @ axis).std())
    print(f'sigma along {args.axis!r} on {Path(args.data).stem}: {sigma:.4f}')
    gz = ds.codes('genre')
    gnames = [str(n) for n in ds.names('genre')]
    cl, jz = (gnames.index('classical'), gnames.index('jazz'))
    gap = float((z[gz == jz].mean(0) - z[gz == cl].mean(0)) @ axis) / sigma
    print(f'classical -> jazz centroid gap: {gap:.2f} sd')
    print(f'alpha=+{args.alpha_max:g} covers {args.alpha_max / gap:.0%} of that gap')
    if args.seeds:
        seeds = np.asarray(args.seeds)
        z0 = z[seeds]
        tags = [
            f'{Path(args.data).stem}[{i}] '
            + ', '.join((f'{f}={ds.names(f)[int(ds.codes(f)[i])]}' for f in ('corpus', 'genre') if f in ds.fields))
            for i in seeds
        ]
    elif args.seed_windows:
        z0, tags = encode_windows(model, args.seed_windows, device)
    else:
        seeds = pick_seeds(z, gz == cl, args.n_seeds)
        z0 = z[seeds]
        tags = [f'{Path(args.data).stem}[{i}]' for i in seeds]
    for t in tags:
        print(f'seed: {t}')
    alphas = np.linspace(-args.alpha_max, args.alpha_max, args.n_alphas)
    trajs = np.stack([np.stack([s + a * sigma * axis for a in alphas]) for s in z0])
    keep = np.concatenate([subsample(np.flatnonzero(gz == g), args.n_per_genre, rng) for g in (cl, jz)])
    mean = z.mean(0)
    zc = z - mean
    cents = np.stack([z[gz == cl].mean(0), z[gz == jz].mean(0)]) - mean
    bases = {
        'pca': (pca_basis(zc), 'PC 1', 'PC 2', 'principal components of the latent'),
        'axis': (
            axis_basis(zc, axis) / np.array([sigma, 1.0]),
            f'{args.axis} axis (alpha, in sd)',
            'top orthogonal PC',
            f'the {args.axis} axis',
        ),
    }
    panels = ('pca', 'axis') if args.panel == 'both' else (args.panel,)
    size = (6.8, 3.0) if len(panels) == 2 else (4.2, 3.2)
    plt.rcParams.update({'font.size': 6.4, 'font.family': 'sans-serif', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axs = plt.subplots(1, len(panels), figsize=size, squeeze=False, constrained_layout=True)
    for ax, key in zip(axs[0], panels):
        basis, xlabel, ylabel, title = bases[key]
        draw(
            ax,
            zc[keep] @ basis,
            gz[keep],
            cents @ basis,
            [(t - mean) @ basis for t in trajs],
            alphas,
            xlabel,
            ylabel,
            title,
        )
    handles = [
        plt.Line2D([], [], ls='', marker='o', ms=4, color=CLASSICAL, label='classical'),
        plt.Line2D([], [], ls='', marker='o', ms=4, color=JAZZ, label='jazz'),
        plt.Line2D([], [], ls='', marker='X', ms=6, color=MUTED, label='genre centroid'),
        plt.Line2D([], [], color=INK, lw=1.1, label='steered seed'),
        plt.Line2D([], [], ls='', marker='v', ms=5, mfc='white', mec=INK, label=f'alpha = {-args.alpha_max:g}'),
        plt.Line2D([], [], ls='', marker='o', ms=5, mfc='white', mec=INK, label='alpha = 0'),
        plt.Line2D([], [], ls='', marker='^', ms=5, mfc='white', mec=INK, label=f'alpha = +{args.alpha_max:g}'),
    ]
    fig.legend(
        handles=handles,
        loc='outside lower center',
        ncol=7 if len(panels) == 2 else 4,
        frameon=False,
        fontsize=5.9,
        handlelength=1.4,
        columnspacing=1.2,
        labelcolor=MUTED,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    png = str(out.with_suffix('.png'))
    fig.savefig(png, dpi=320, bbox_inches='tight')
    print(f'wrote {out} and {png}')


if __name__ == '__main__':
    main()

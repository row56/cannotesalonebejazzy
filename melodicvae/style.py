from dataclasses import dataclass, field
import numpy as np
import torch


@torch.no_grad()
def encode_dataset(model, loader, device: str = "cpu"):
    model.eval().to(device)
    zs, idx = ([], [])
    for x, i in loader:
        mu, _ = model.encode(x.to(device))
        zs.append(mu.cpu().numpy())
        idx.append(np.asarray(i))
    z = np.concatenate(zs)
    order = np.argsort(np.concatenate(idx))
    return z[order]


@dataclass
class StyleSpace:
    label_names: list[str]
    centroids: np.ndarray
    mean: np.ndarray
    axes: dict = field(default_factory=dict)
    counts: list = field(default_factory=list)

    @classmethod
    def fit(cls, z: np.ndarray, y: np.ndarray, label_names: list[str], min_count: int = 100):
        mean = z.mean(0)
        centroids, counts = ([], [])
        for i in range(len(label_names)):
            mask = y == i
            counts.append(int(mask.sum()))
            centroids.append(z[mask].mean(0))
        centroids = np.stack(centroids)
        space = cls(label_names, centroids, mean, counts=counts)
        for i, name in enumerate(label_names):
            if counts[i] >= min_count:
                space.axes[name] = _unit(centroids[i] - mean)
        return space

    def attribute_axis(self, name: str, z: np.ndarray, values: np.ndarray, ridge: float = 0.001):
        zc = z - self.mean
        v = np.asarray(values, dtype=float)
        v = (v - v.mean()) / (v.std() + 1e-08)
        gram = zc.T @ zc + ridge * len(zc) * np.eye(zc.shape[1])
        w = np.linalg.solve(gram, zc.T @ v)
        self.axes[name] = _unit(w)
        return self.axes[name]

    def contrast_axis(self, name: str, positive: str, negative: str):
        i, j = (self.label_names.index(positive), self.label_names.index(negative))
        axis = _unit(self.centroids[i] - self.centroids[j])
        self.axes[name] = axis
        return axis

    def steer(self, z: np.ndarray, axis: str, alpha: float):
        return z + alpha * self.axes[axis]

    def save(self, path):
        np.savez(
            path,
            label_names=np.array(self.label_names, dtype=object),
            centroids=self.centroids,
            mean=self.mean,
            counts=np.array(self.counts, dtype=np.int64),
            axis_names=np.array(list(self.axes), dtype=object),
            axis_vectors=np.stack(list(self.axes.values())) if self.axes else np.zeros((0, 1)),
        )

    @classmethod
    def load(cls, path):
        b = np.load(path, allow_pickle=True)
        space = cls(list(b["label_names"]), b["centroids"], b["mean"])
        space.axes = {n: v for n, v in zip(b["axis_names"], b["axis_vectors"])}
        space.counts = b["counts"].tolist() if "counts" in b.files else []
        return space


def _unit(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _midranks(v: np.ndarray):
    order = np.argsort(v, kind="mergesort")
    s = v[order]
    fresh = np.concatenate([[True], s[1:] != s[:-1]])
    start = np.flatnonzero(fresh)
    end = np.concatenate([start[1:], [len(s)]])
    group = np.cumsum(fresh) - 1
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = 0.5 * (start[group] + end[group] - 1) + 1.0
    return ranks


def auc(pos: np.ndarray, neg: np.ndarray):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _midranks(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def axis_auc(z: np.ndarray, y: np.ndarray, axis: np.ndarray, label_idx: int):
    scores = z @ axis
    return auc(scores[y == label_idx], scores[y != label_idx])


def descriptor_correlation(z: np.ndarray, descriptors: np.ndarray, axis: np.ndarray):
    proj = z @ axis
    proj = (proj - proj.mean()) / (proj.std() + 1e-08)
    d = (descriptors - descriptors.mean(0)) / (descriptors.std(0) + 1e-08)
    return (proj[:, None] * d).mean(0)

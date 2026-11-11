import numpy as np
from .repr import ONSET_OFFSET, REST

MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11])


def _onsets(tokens: np.ndarray):
    tokens = np.asarray(tokens).ravel()
    idx = np.flatnonzero(tokens >= ONSET_OFFSET)
    return (idx, tokens[idx] - ONSET_OFFSET)


def note_density(tokens: np.ndarray):
    idx, _ = _onsets(tokens)
    return len(idx) / max(len(np.asarray(tokens).ravel()), 1)


def pitch_range(tokens: np.ndarray):
    _, p = _onsets(tokens)
    return float(p.max() - p.min()) if len(p) > 1 else 0.0


def mean_interval(tokens: np.ndarray):
    _, p = _onsets(tokens)
    return float(np.abs(np.diff(p)).mean()) if len(p) > 1 else 0.0


def leap_rate(tokens: np.ndarray, threshold: int = 7):
    _, p = _onsets(tokens)
    if len(p) < 2:
        return 0.0
    return float((np.abs(np.diff(p)) > threshold).mean())


def rest_ratio(tokens: np.ndarray):
    t = np.asarray(tokens).ravel()
    return float((t == REST).mean())


def syncopation(tokens: np.ndarray, steps_per_beat: int = 4):
    idx, _ = _onsets(tokens)
    return float((idx % steps_per_beat != 0).mean()) if len(idx) else 0.0


def metric_weights(n_steps: int, steps_per_bar: int = 16):
    levels = int(np.log2(steps_per_bar))
    w = np.empty(n_steps, dtype=int)
    for i in range(n_steps):
        pos = i % steps_per_bar
        if pos == 0:
            w[i] = 0
            continue
        div = 1
        while pos % (div * 2) == 0:
            div *= 2
        w[i] = -(levels - int(np.log2(div)))
    return w


def lhl_syncopation(tokens: np.ndarray, steps_per_bar: int = 16, normalise: bool = True):
    t = np.asarray(tokens).ravel()
    n = len(t)
    if n == 0:
        return 0.0
    w = metric_weights(n, steps_per_bar)
    onsets = np.flatnonzero(t >= ONSET_OFFSET)
    if not len(onsets):
        return 0.0
    total = 0
    for k, start in enumerate(onsets):
        end = onsets[k + 1] if k + 1 < len(onsets) else n
        for j in range(start + 1, end):
            if t[j] == REST:
                break
            if w[j] > w[start]:
                total += w[j] - w[start]
    return float(total / len(onsets)) if normalise else float(total)


def beat_vacancy(tokens: np.ndarray, steps_per_beat: int = 4):
    idx, _ = _onsets(tokens)
    n_steps = len(np.asarray(tokens).ravel())
    beats = np.arange(0, n_steps, steps_per_beat)
    if not len(beats):
        return 0.0
    return float(1.0 - np.isin(beats, idx).sum() / len(beats))


def best_scale(tokens: np.ndarray):
    _, p = _onsets(tokens)
    if len(p) == 0:
        return (MAJOR_SCALE.copy(), 0.0)
    hist = np.bincount(p % 12, minlength=12)
    fits = [hist[(MAJOR_SCALE + root) % 12].sum() for root in range(12)]
    root = int(np.argmax(fits))
    return ((MAJOR_SCALE + root) % 12, float(max(fits) / len(p)))


def chromaticism(tokens: np.ndarray):
    _, p = _onsets(tokens)
    if len(p) == 0:
        return 0.0
    return 1.0 - best_scale(tokens)[1]


def pitch_class_entropy(tokens: np.ndarray):
    _, p = _onsets(tokens)
    if len(p) == 0:
        return 0.0
    hist = np.bincount(p % 12, minlength=12).astype(float)
    hist /= hist.sum()
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())


DESCRIPTORS = {
    "note_density": note_density,
    "pitch_range": pitch_range,
    "mean_interval": mean_interval,
    "rest_ratio": rest_ratio,
    "syncopation": syncopation,
    "chromaticism": chromaticism,
    "pc_entropy": pitch_class_entropy,
    "beat_vacancy": beat_vacancy,
    "lhl_syncopation": lhl_syncopation,
}


def describe(tokens: np.ndarray):
    return {name: fn(tokens) for name, fn in DESCRIPTORS.items()}


def describe_batch(batch: np.ndarray):
    return np.array([[fn(w) for fn in DESCRIPTORS.values()] for w in batch])

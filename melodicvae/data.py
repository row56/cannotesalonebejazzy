import copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .labels import LabelSet
from .repr import ONSET_OFFSET, REST, STEPS_PER_BAR, skyline, transpose


def beat_grid(pm, steps_per_beat: int = 4):
    beats = pm.get_beats()
    if len(beats) < 2:
        return None
    sub = np.arange(steps_per_beat) / steps_per_beat
    grid = (beats[:-1, None] + np.diff(beats)[:, None] * sub[None, :]).ravel()
    return np.append(grid, beats[-1])


def midi_to_tokens(path: str | Path, steps_per_beat: int = 4):
    import pretty_midi

    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None
    if not pm.instruments:
        return None
    grid = beat_grid(pm, steps_per_beat)
    if grid is None or len(grid) < STEPS_PER_BAR:
        return None
    notes = [
        (np.searchsorted(grid, n.start), np.searchsorted(grid, n.end), n.pitch)
        for inst in pm.instruments
        if not inst.is_drum
        for n in inst.notes
    ]
    return skyline(notes, n_steps=len(grid))


def window(tokens: np.ndarray, seq_len: int, hop: int, min_onsets: int = 8, min_pitches: int = 3):
    out = []
    for start in range(0, max(len(tokens) - seq_len + 1, 0), hop):
        w = tokens[start : start + seq_len]
        onsets = w[w >= ONSET_OFFSET]
        if len(onsets) < min_onsets or len(np.unique(onsets)) < min_pitches:
            continue
        if (w == REST).mean() > 0.6:
            continue
        out.append(w.astype(np.int16))
    return out


def grouped_split(groups, val_frac: float, seed: int = 0):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = min(max(int(round(len(uniq) * val_frac)), 1), len(uniq) - 1)
    is_val = np.isin(groups, uniq[:n_val])
    return (np.flatnonzero(~is_val), np.flatnonzero(is_val))


class MelodyDataset(Dataset):

    def __init__(self, npz_path: str | Path, augment_transpose: int = 0):
        blob = np.load(npz_path, allow_pickle=True)
        self.x = blob["x"]
        self.labels = LabelSet.from_npz(blob)
        self.source_file = blob["source_file"] if "source_file" in blob.files else None
        self.augment = augment_transpose

    def eval_view(self):
        other = copy.copy(self)
        other.augment = 0
        return other

    @property
    def fields(self):
        return self.labels.fields

    def names(self, field: str):
        return self.labels.names[field]

    def codes(self, field: str):
        return self.labels.codes[field]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i: int):
        w = self.x[i]
        if self.augment:
            shift = np.random.randint(-self.augment, self.augment + 1)
            if shift:
                shifted = transpose(w, shift)
                if shifted is not None:
                    w = shifted
        return (torch.from_numpy(w.astype(np.int64)), i)

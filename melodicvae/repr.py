import numpy as np

REST = 0
HOLD = 1
ONSET_OFFSET = 2
VOCAB_SIZE = 130
STEPS_PER_BAR = 16


def pitch_to_token(pitch: int):
    return ONSET_OFFSET + int(pitch)


def token_to_pitch(token: int):
    return int(token) - ONSET_OFFSET if token >= ONSET_OFFSET else None


def skyline(notes, n_steps: int):
    top_pitch = np.full(n_steps, -1, dtype=np.int16)
    note_spans = []
    for start, end, pitch in notes:
        start = int(start)
        if start >= n_steps or start < 0:
            continue
        end = min(max(int(end), start + 1), n_steps)
        np.maximum(top_pitch[start:end], pitch, out=top_pitch[start:end])
        note_spans.append((start, pitch))
    sounding = top_pitch >= 0
    changed = np.empty(n_steps, dtype=bool)
    changed[0] = sounding[0]
    changed[1:] = sounding[1:] & (top_pitch[1:] != top_pitch[:-1])
    for start, pitch in note_spans:
        if top_pitch[start] == pitch:
            changed[start] = True
    tokens = np.full(n_steps, REST, dtype=np.int16)
    tokens[sounding] = HOLD
    starts = changed & sounding
    tokens[starts] = top_pitch[starts] + ONSET_OFFSET
    return tokens


def tokens_to_midi(tokens: np.ndarray, grid_hz: float, velocity: int = 90, program: int = 0):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    cur_pitch, cur_start = (None, None)

    def close(step: int):
        nonlocal cur_pitch, cur_start
        if cur_pitch is not None:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=cur_pitch,
                    start=cur_start / grid_hz,
                    end=step / grid_hz,
                )
            )
            cur_pitch, cur_start = (None, None)

    for step, tok in enumerate(np.asarray(tokens).ravel()):
        tok = int(tok)
        if tok == HOLD:
            if cur_pitch is None:
                continue
        elif tok == REST:
            close(step)
        else:
            close(step)
            cur_pitch, cur_start = (token_to_pitch(tok), step)
    close(len(tokens))
    pm.instruments.append(inst)
    return pm


KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_tonic(tokens: np.ndarray):
    t = np.asarray(tokens).ravel()
    pitches = t[t >= ONSET_OFFSET] - ONSET_OFFSET
    if len(pitches) < 8:
        return None
    hist = np.bincount(pitches % 12, minlength=12).astype(float)
    hist -= hist.mean()
    if not hist.any():
        return None
    best_root, best_score = (0, -np.inf)
    for profile in (KK_MAJOR, KK_MINOR):
        p = profile - profile.mean()
        for root in range(12):
            score = float(hist @ np.roll(p, root))
            if score > best_score:
                best_root, best_score = (root, score)
    return best_root


def normalise_key(tokens: np.ndarray):
    tonic = estimate_tonic(tokens)
    if tonic is None:
        return tokens
    shift = -tonic if tonic <= 6 else 12 - tonic
    out = transpose(tokens, shift)
    return tokens if out is None else out


def centre_octave(tokens: np.ndarray, target: int = 67):
    pitches = tokens[tokens >= ONSET_OFFSET] - ONSET_OFFSET
    if not len(pitches):
        return tokens
    shift = 12 * int(round((target - np.median(pitches)) / 12))
    out = transpose(tokens, shift)
    return tokens if out is None else out


def transpose(tokens: np.ndarray, semitones: int):
    out = np.array(tokens, dtype=np.int16)
    onsets = out >= ONSET_OFFSET
    if not onsets.any():
        return out
    pitches = out[onsets] - ONSET_OFFSET + semitones
    if pitches.min() < 0 or pitches.max() > 127:
        return None
    out[onsets] = pitches + ONSET_OFFSET
    return out

import glob, sys, wave
import numpy as np

files = sorted(glob.glob(sys.argv[1] + '/**/*.wav', recursive=True))
data = {}
for f in files:
    w = wave.open(f)
    p = w.getparams()
    a = np.frombuffer(w.readframes(p.nframes), np.int16).astype(np.float32)
    data[f] = (a.reshape(-1, p.nchannels).mean(1), p.framerate)
    w.close()
g = 10 ** (-1 / 20) * 32767 / max((np.abs(a).max() for a, _ in data.values()))
for f, (a, sr) in data.items():
    o = wave.open(f, 'wb')
    o.setnchannels(1)
    o.setsampwidth(2)
    o.setframerate(sr)
    o.writeframes(np.clip(a * g, -32768, 32767).astype(np.int16).tobytes())
    o.close()
print(f'{len(files)} files, shared gain x{g:.1f}')

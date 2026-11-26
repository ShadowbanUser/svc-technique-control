"""Small acoustic proxies used by the feasibility probe. NumPy only.

These are screening numbers, not perceptual evidence:

* CPP (cepstral peak prominence, dB): the standard acoustic correlate of
  breathiness. Breathier phonation has a weaker harmonic structure, so CPP is
  LOWER.
* HF ratio (dB): energy in 4-8 kHz relative to 0.1-4 kHz, at 16 kHz sampling.
  Aspiration noise raises it.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 640      # 40 ms
HOP = 160        # 10 ms, matches the RMVPE frame rate
NFFT = 1024


def _frames(wave):
    wave = np.asarray(wave, dtype=np.float64)
    if len(wave) < FRAME:
        wave = np.pad(wave, (0, FRAME - len(wave)))
    count = 1 + (len(wave) - FRAME) // HOP
    index = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    return wave[index]


def cpp_track(wave_16k, fmin=70.0, fmax=1100.0):
    """Per-frame CPP in dB (NaN for near-silent frames)."""
    frames = _frames(wave_16k)
    rms = np.sqrt((frames ** 2).mean(1))
    window = np.hanning(FRAME)
    spectrum = np.abs(np.fft.rfft(frames * window, NFFT, axis=1))
    log_spectrum = 20.0 * np.log10(spectrum + 1e-10)
    cepstrum = np.abs(np.fft.irfft(log_spectrum, axis=1))[:, :NFFT // 2]
    cepstrum_db = 20.0 * np.log10(cepstrum + 1e-10)
    quefrency = np.arange(NFFT // 2) / SR
    low, high = int(SR / fmax), int(SR / fmin)
    fit_from = int(0.001 * SR)
    q_fit = quefrency[fit_from:high + 1]
    out = np.full(len(frames), np.nan)
    design = np.stack([q_fit, np.ones_like(q_fit)], axis=1)
    pinv = np.linalg.pinv(design)
    for i in range(len(frames)):
        if rms[i] < 1e-4:
            continue
        slope, intercept = pinv @ cepstrum_db[i, fit_from:high + 1]
        peak = low + int(np.argmax(cepstrum_db[i, low:high + 1]))
        out[i] = cepstrum_db[i, peak] - (slope * quefrency[peak] + intercept)
    return out


def hf_ratio_track(wave_16k):
    """Per-frame 4-8 kHz vs 0.1-4 kHz energy ratio in dB."""
    frames = _frames(wave_16k)
    power = np.abs(np.fft.rfft(frames * np.hanning(FRAME), NFFT, axis=1)) ** 2
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    high = power[:, (freqs >= 4000) & (freqs <= 8000)].sum(1)
    low = power[:, (freqs >= 100) & (freqs < 4000)].sum(1)
    return 10.0 * np.log10((high + 1e-12) / (low + 1e-12))


def _align(track, voiced):
    n = min(len(track), len(voiced))
    return track[:n], np.asarray(voiced[:n], dtype=bool)


def summarize(wave_16k, f0=None):
    """f0: optional 100 fps F0 track (Hz, 0 = unvoiced) used to pick voiced frames."""
    wave_16k = np.asarray(wave_16k, dtype=np.float64)
    cpp = cpp_track(wave_16k)
    hf = hf_ratio_track(wave_16k)
    if f0 is not None:
        voiced = np.asarray(f0) > 1.0
    else:
        frames = _frames(wave_16k)
        rms = np.sqrt((frames ** 2).mean(1))
        voiced = rms > 0.1 * np.nanmax(rms)
    cpp_v, mask = _align(cpp, voiced)
    hf_v, _ = _align(hf, voiced)
    keep = mask & np.isfinite(cpp_v)
    return {
        "cpp_db": float(np.mean(cpp_v[keep])) if keep.any() else None,
        "hf_ratio_db": float(np.mean(hf_v[keep])) if keep.any() else None,
        "voiced_fraction": float(mask.mean()) if len(mask) else None,
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(wave_16k ** 2)) + 1e-12)),
        "peak": float(np.max(np.abs(wave_16k))) if len(wave_16k) else None,
    }


def f0_agreement(f0_out, f0_src):
    """Compare two 100 fps F0 tracks on frames voiced in both."""
    n = min(len(f0_out), len(f0_src))
    a, b = np.asarray(f0_out[:n], dtype=np.float64), np.asarray(f0_src[:n], dtype=np.float64)
    both = (a > 1.0) & (b > 1.0)
    if both.sum() < 10:
        return {"f0_common_frames": int(both.sum()), "f0_corr": None, "f0_mae_cents": None,
                "voicing_agreement": float(((a > 1) == (b > 1)).mean()) if n else None}
    la, lb = np.log2(a[both]), np.log2(b[both])
    corr = float(np.corrcoef(la, lb)[0, 1]) if la.std() > 0 and lb.std() > 0 else None
    return {"f0_common_frames": int(both.sum()), "f0_corr": corr,
            "f0_mae_cents": float(np.mean(np.abs(la - lb)) * 1200.0),
            "voicing_agreement": float(((a > 1) == (b > 1)).mean())}


def logmel_distance(wave_a_16k, wave_b_16k, n_mels=80):
    """Mean absolute difference (dB) between two log-mel spectrograms, frame-aligned from the start."""
    import librosa
    def logmel(wave):
        mel = librosa.feature.melspectrogram(y=np.asarray(wave, dtype=np.float32), sr=SR, n_fft=NFFT,
                                             hop_length=HOP, n_mels=n_mels, power=2.0)
        return 10.0 * np.log10(mel + 1e-10)
    a, b = logmel(wave_a_16k), logmel(wave_b_16k)
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        return None
    floor = max(a.max(), b.max()) - 80.0
    return float(np.mean(np.abs(np.maximum(a[:, :n], floor) - np.maximum(b[:, :n], floor))))

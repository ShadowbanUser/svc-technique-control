"""Objective measures shared by experiment 06. NumPy / SciPy / librosa only.

Spectral-envelope representation
--------------------------------
44.1 kHz audio, 2048-point STFT, 10 ms hop (100 frames/s), 128 mel bands 50 Hz - 16 kHz,
power in dB, floored 80 dB below the utterance maximum. An orthonormal DCT over the mel
axis gives cepstra; coefficients 1..30 are kept (c0, the overall level, is dropped), which
is a smoothed spectral envelope without fine harmonic detail.

Envelope distance between two frame-aligned sequences A, B (T x 30):

    D = sqrt( mean_t ||A_t - B_t||^2 / 128 )        (dB, RMS over mel bands)

Because the DCT is orthonormal, D is the RMS dB difference between the two cepstrally
smoothed log-mel spectra, level removed.

CMN variant: subtract each sequence's own mean over the compared frames first. A fixed
per-band gain (an idealised static equaliser) shifts every frame by the same vector, so it
cannot change D_cmn; and for any sequence, D_cmn(A, B) is exactly the smallest D that
adding one fixed vector to A can reach. D_cmn(base, real) is therefore the best any
idealised static filter on the base output can do.
"""
from __future__ import annotations

import numpy as np

SR = 44100
N_FFT = 2048
HOP = 441
N_MELS = 128
FMIN, FMAX = 50.0, 16000.0
N_CEP = 30
FLOOR_DB = 80.0
ACTIVE_DB = 35.0
N_BROAD = 8


def logmel_db(wave44):
    import librosa
    mel = librosa.feature.melspectrogram(y=np.asarray(wave44, dtype=np.float32), sr=SR, n_fft=N_FFT,
                                         hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    db = 10.0 * np.log10(mel + 1e-10)
    return np.maximum(db, db.max() - FLOOR_DB)          # (bands, frames)


def active_frames(lm_db, below=ACTIVE_DB):
    """Frames whose band-averaged dB level is within `below` dB of the loudest frame.

    Averaging dB values (not power) means a fixed per-band gain moves every frame by the
    same amount, so an equaliser does not change which frames count as active."""
    level = lm_db.mean(axis=0)
    return level > level.max() - below


def cepstra(lm_db):
    from scipy.fft import dct
    return dct(lm_db, type=2, norm="ortho", axis=0)[1:N_CEP + 1].T   # (frames, N_CEP)


def analyse(wave44):
    lm = logmel_db(wave44)
    return {"cep": cepstra(lm), "active": active_frames(lm), "frames": lm.shape[1]}


def envelope_distance(a, b, cmn=False):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0:
        return None
    if cmn:
        a = a - a.mean(0)
        b = b - b.mean(0)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)) / N_MELS))


def variance_ratio(a, b):
    """Mean over coefficients of var(a)/var(b): below 1 means a is smoother over time than b."""
    va, vb = np.var(a, axis=0), np.var(b, axis=0)
    keep = vb > 1e-9
    return float(np.mean(va[keep] / vb[keep])) if keep.any() else None


def map_100fps(mapping50, n_real, n_out):
    """mapping50[j] = source frame (50 Hz) for real-take frame j (50 Hz), as produced by the
    DTW in prep_technique_pairs.align. Returns, for each real-take frame at 100 fps, the
    frame index on the source (= output) time line at 100 fps."""
    mapping50 = np.asarray(mapping50, dtype=np.int64)
    k = np.arange(n_real)
    idx = mapping50[np.minimum(k // 2, len(mapping50) - 1)] * 2 + (k % 2)
    return np.clip(idx, 0, n_out - 1)


def compare(out, real, index=None):
    """Distances between an analysed output and an analysed real take.

    index: for each real frame, the output frame it corresponds to (None = same time line).
    Only frames active in the real take are compared."""
    n = real["frames"]
    if index is None:
        index = np.minimum(np.arange(n), out["frames"] - 1)
    keep = real["active"][:n]
    a, b = out["cep"][index[keep]], real["cep"][:n][keep]
    return {"d_raw": envelope_distance(a, b), "d_cmn": envelope_distance(a, b, cmn=True),
            "var_ratio": variance_ratio(a, b), "frames_compared": int(keep.sum())}


# ---------------------------------------------------------------- classifier features

def _cpp_stats(wave44):
    import librosa
    from steer.metrics import cpp_track, _frames
    w16 = librosa.resample(np.asarray(wave44, dtype=np.float32), orig_sr=SR, target_sr=16000)
    cpp = cpp_track(w16)
    rms = np.sqrt((_frames(w16) ** 2).mean(1))
    keep = np.isfinite(cpp) & (rms > rms.max() * 10 ** (-ACTIVE_DB / 20))
    return (float(np.mean(cpp[keep])), float(np.std(cpp[keep]))) if keep.any() else (0.0, 0.0)


def feature_names():
    dyn = ([f"cep_std_{i}" for i in range(1, N_CEP + 1)] + [f"cep_dabs_{i}" for i in range(1, N_CEP + 1)]
           + [f"band_std_{i}" for i in range(N_BROAD)] + ["cpp_mean", "cpp_std"])
    return {"dyn": dyn, "full": dyn + [f"cep_mean_{i}" for i in range(1, N_CEP + 1)]}


def technique_features(wave44):
    """Two feature sets for the technique classifier.

    dyn  : how the spectrum changes over time (std and mean absolute frame-to-frame change of
           cepstra, std of 8 broad-band levels) plus CPP. A fixed per-band gain leaves all of
           these unchanged (CPP approximately), so an equaliser cannot move them.
    full : dyn plus the mean cepstrum, i.e. the static spectral shape an equaliser does set.
    Overall level (c0) is in neither set."""
    lm = logmel_db(wave44)
    act = active_frames(lm)
    if act.sum() < 5:
        act = np.ones_like(act)
    cep = cepstra(lm)
    c = cep[act]
    pair = act[1:] & act[:-1]
    dabs = np.abs(np.diff(cep, axis=0))[pair] if pair.any() else np.zeros((1, N_CEP))
    broad = lm.reshape(N_BROAD, N_MELS // N_BROAD, -1).mean(axis=1).T[act]
    cpp_mean, cpp_std = _cpp_stats(wave44)
    dyn = np.concatenate([c.std(0), dabs.mean(0), broad.std(0), [cpp_mean, cpp_std]])
    full = np.concatenate([dyn, c.mean(0)])
    return {"dyn": dyn.astype(np.float64), "full": full.astype(np.float64)}


# ---------------------------------------------------------------- logistic regression

def fit_logreg(x, y, n_classes, l2, weights=None, iters=200):
    """Multinomial logistic regression, L2 on weights (not bias), full-batch L-BFGS, float64.
    x is expected standardised. Returns (W, b) as NumPy arrays."""
    import torch
    torch.manual_seed(0)
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.long)
    wt = torch.ones(len(y), dtype=torch.float64) if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    W = torch.zeros(x.shape[1], n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        nll = torch.nn.functional.cross_entropy(xt @ W + b, yt, reduction="none")
        loss = (wt * nll).sum() / wt.sum() + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def predict_proba(x, W, b):
    z = x @ W + b
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def balanced_accuracy(y, pred, n_classes):
    rec = [np.mean(pred[y == k] == k) for k in range(n_classes) if np.any(y == k)]
    return float(np.mean(rec)) if rec else None

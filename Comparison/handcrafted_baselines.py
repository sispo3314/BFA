import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SVM_C = 1.0
RF_TREES = 300


def handcrafted_features(x, fs):
    """Window-level handcrafted features shared by SVM-RBF and Random Forest.

    x:  [N, T, C] normalized inertial windows
    fs: sampling rate in Hz
    returns [N, n_features]

    Time-Domain (10, per channel)
      - Mean, Standard Deviation (SD), Median, Interquartile Range (IQR),
        Min, Max, Root Mean Square (RMS), Mean Square (Energy),
        Skewness, Kurtosis

    Frequency-Domain (3, per channel)
      - Dominant Frequency, Spectral Entropy, Spectral Centroid

    Cross-Axis (4, per tri-axial sensor)
      - Signal Magnitude Area (SMA), Pairwise Axis Correlation (x-y, x-z, y-z)
    """
    x = np.asarray(x, dtype=np.float32)

    # ------------------------------------------------------------------
    # Time-Domain (10)
    # ------------------------------------------------------------------
    mean = x.mean(axis=1)                                             # Mean
    sd = x.std(axis=1)                                                # Standard Deviation (SD)
    median = np.median(x, axis=1)                                     # Median
    iqr = np.quantile(x, .75, axis=1) - np.quantile(x, .25, axis=1)   # Interquartile Range (IQR)
    x_min = x.min(axis=1)                                             # Min
    x_max = x.max(axis=1)                                             # Max
    rms = np.sqrt((x * x).mean(axis=1))                               # Root Mean Square (RMS)
    energy = (x * x).mean(axis=1)                                     # Mean Square (Energy)

    centered = x - mean[:, None]
    m2 = np.maximum((centered ** 2).mean(axis=1), 1e-12)
    skew = (centered ** 3).mean(axis=1) / (m2 ** 1.5)                 # Skewness
    kurt = (centered ** 4).mean(axis=1) / (m2 ** 2) - 3.0             # Kurtosis

    # ------------------------------------------------------------------
    # Frequency-Domain (3)
    # ------------------------------------------------------------------
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freq = np.fft.rfftfreq(x.shape[1], d=1.0 / fs)
    power[:, 0] = 0.0                                                 # remove DC component
    den = power.sum(axis=1) + 1e-12
    prob = power / den[:, None]

    dominant = freq[power.argmax(axis=1)]                             # Dominant Frequency
    entropy = -(prob * np.log(prob + 1e-12)).sum(axis=1)              # Spectral Entropy
    centroid = (power * freq[None, :, None]).sum(axis=1) / den        # Spectral Centroid

    feats = [mean, sd, median, iqr, x_min, x_max, rms, energy, skew, kurt,
             dominant, entropy, centroid]

    # ------------------------------------------------------------------
    # Cross-Axis (4)
    # ------------------------------------------------------------------
    for start in range(0, x.shape[2] - 2, 3):
        tri = x[:, :, start:start + 3]

        # Signal Magnitude Area (SMA)
        feats.append(np.abs(tri).sum(axis=2).mean(axis=1, keepdims=True))

        # Pairwise Axis Correlation (x-y, x-z, y-z)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            ca = tri[:, :, a] - tri[:, :, a].mean(axis=1, keepdims=True)
            cb = tri[:, :, b] - tri[:, :, b].mean(axis=1, keepdims=True)
            corr = (ca * cb).sum(axis=1) / np.sqrt(
                np.maximum((ca * ca).sum(axis=1) * (cb * cb).sum(axis=1), 1e-12))
            feats.append(corr[:, None])

    return np.nan_to_num(np.concatenate(feats, axis=1),
                         nan=0, posinf=0, neginf=0).astype(np.float32)


def build_svm_rbf(c=SVM_C):
    """StandardScaler (fit on training windows only) + RBF-kernel SVM."""
    return make_pipeline(
        StandardScaler(),
        SVC(C=c, kernel="rbf", gamma="scale", cache_size=512),
    )


def build_random_forest(n_trees=RF_TREES, seed=42, n_jobs=1):
    """StandardScaler (fit on training windows only) + Random Forest."""
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=n_jobs),
    )

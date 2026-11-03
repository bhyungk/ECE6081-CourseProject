"""The deliberately small, fixed feature set used by the teaching baseline.

All calculations use the samples within one window. No participant-level or
dataset-level normalization is performed here.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("mean", "std", "min", "max", "median", "iqr", "slope")
EDA_FEATURES = tuple(f"eda_{name}" for name in FEATURE_NAMES)
BVP_FEATURES = tuple(f"bvp_{name}" for name in FEATURE_NAMES)


def extract_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    """Calculate seven statistics, with slope measured per second.

    ``std`` is the population standard deviation (``ddof=0``). The IQR uses
    NumPy's linear percentile interpolation. Slope is the ordinary least
    squares coefficient for sample times ``arange(n) / sample_rate``.

    A window with fewer than two samples, or any nonfinite sample, returns
    seven NaNs. The calling pipeline keeps its window and learns imputation
    values using training participants only.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional or have shape (n, 1)")
    missing = {name: float("nan") for name in FEATURE_NAMES}
    if values.size < 2 or not np.isfinite(values).all():
        return missing

    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values))
        times = np.arange(values.size, dtype=np.float64) / float(sample_rate)
        centered_times = times - times.mean()
        slope = float(
            np.dot(centered_times, values - mean)
            / np.dot(centered_times, centered_times)
        )
        q25, q75 = np.percentile(values, [25, 75], method="linear")
        result = {
            "mean": mean,
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "iqr": float(q75 - q25),
            "slope": slope,
        }
    # Also make exceptional arithmetic overflow an explicit missing-feature
    # event rather than allowing infinity to reach the estimator.
    return result if np.isfinite(list(result.values())).all() else missing

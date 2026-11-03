import numpy as np
import pytest

from wesad_starter.features import FEATURE_NAMES, extract_features


def test_fixed_statistics_and_slope_units():
    # x(t) = 3 + 2*t at 4 Hz: slope must be 2 units/second, not 0.5/sample.
    signal = 3 + 2 * np.arange(240) / 4
    result = extract_features(signal, 4)
    assert tuple(result) == FEATURE_NAMES
    assert result["mean"] == pytest.approx(62.75)
    assert result["std"] == pytest.approx(np.std(signal, ddof=0))
    assert result["min"] == pytest.approx(3)
    assert result["max"] == pytest.approx(122.5)
    assert result["median"] == pytest.approx(62.75)
    assert result["iqr"] == pytest.approx(59.75)
    assert result["slope"] == pytest.approx(2)


def test_constant_window_is_valid():
    result = extract_features(np.full(240, 7.5), 4)
    assert result["mean"] == 7.5
    assert result["std"] == result["iqr"] == result["slope"] == 0
    assert all(np.isfinite(value) for value in result.values())


@pytest.mark.parametrize("signal", [[], [1], [1, np.nan], [1, np.inf]])
def test_invalid_window_is_missing_as_a_whole(signal):
    assert all(np.isnan(value) for value in extract_features(np.asarray(signal), 4).values())


def test_sample_rate_is_required_to_be_positive():
    with pytest.raises(ValueError, match="positive"):
        extract_features(np.arange(3), 0)

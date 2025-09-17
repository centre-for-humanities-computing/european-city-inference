import pytest
import numpy as np
from voting import kl_divergence  # Make sure your function is importable

def test_kl_divergence():
    # Test case 1: Valid values
    mean_belief_1 = np.array(0.0)
    precision_belief_1 = np.array(1.0)  # variance = 1.0
    mean_pref_1 = np.array(1.0)
    precision_pref_1 = np.array(1.0)    # variance = 1.0
    result_1 = kl_divergence(mean_belief_1, precision_belief_1, mean_pref_1, precision_pref_1)
    expected_1 = np.array(0.5)
    assert np.allclose(result_1, expected_1, atol=1e-10)

    # Test case 2: Equal variances
    mean_belief_2 = np.array(0.0)
    precision_belief_2 = np.array(1.0)  # variance = 1.0
    mean_pref_2 = np.array(0.0)
    precision_pref_2 = np.array(1.0)    # variance = 1.0
    result_2 = kl_divergence(mean_belief_2, precision_belief_2, mean_pref_2, precision_pref_2)
    expected_2 = np.array(0.0)
    assert np.allclose(result_2, expected_2, atol=1e-10)

    # Test case 3: Different variances
    mean_belief_3 = np.array(1.0)
    precision_belief_3 = np.array(0.5)  # variance = 2.0
    mean_pref_3 = np.array(0.0)
    precision_pref_3 = np.array(2.0)    # variance = 0.5
    result_3 = kl_divergence(mean_belief_3, precision_belief_3, mean_pref_3, precision_pref_3)
    expected_3 = np.log(np.sqrt(0.5) / np.sqrt(2.0)) + (2.0 + (1.0 - 0.0)**2) / (2 * 0.5) - 0.5
    assert np.allclose(result_3, expected_3, atol=1e-10)

    # Test case 4: Variances near zero
    with pytest.raises(ValueError, match="Variances must be positive."):
        mean_belief_4 = np.array(0.0)
        precision_belief_4 = np.array(1e-10)  # very large variance
        mean_pref_4 = np.array(0.0)
        precision_pref_4 = np.array(1.0)
        kl_divergence(mean_belief_4, precision_belief_4, mean_pref_4, precision_pref_4)

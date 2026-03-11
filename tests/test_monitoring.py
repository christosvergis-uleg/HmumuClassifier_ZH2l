import numpy as np
from eventclf.monitoring import feature_summary, psi

def test_feature_summary():
    x = np.array([1.0, 2.0, np.nan])
    s = feature_summary(x)
    assert s["n"] == 3
    assert s["finite_n"] == 2

def test_psi_small_for_same():
    a = np.random.default_rng(0).normal(size=1000)
    b = a.copy()
    assert psi(a, b) < 1e-6
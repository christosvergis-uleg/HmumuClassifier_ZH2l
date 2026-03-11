import numpy as np
from eventclf.eval import evaluate_binary_classifier

def test_metrics_runs():
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.8, 0.2, 0.7, 0.6])
    m = evaluate_binary_classifier(y, s)
    assert "auc" in m and m["n"] == 5

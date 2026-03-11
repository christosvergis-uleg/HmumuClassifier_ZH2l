import json
from pathlib import Path

from eventclf.utils.io import load_json

BASELINE = Path("tests/baselines/xgb_hmumu_zh2l_metrics.json")

def test_auc_not_regressed():
    baseline = load_json(BASELINE)
    current = load_json("artifacts/metrics_oof.json")  # produced by scripts/train_cv.py

    # allow small wiggle room (floating + small data shifts)
    assert current["auc"] >= baseline["auc"] - 0.005
    assert current["logloss"] <= baseline["logloss"] + 0.01
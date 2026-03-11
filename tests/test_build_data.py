import pandas as pd
import numpy as np
from eventclf.data import DatasetSpec, build_arrays

def test_build_arrays_basic():
    df = pd.DataFrame({
        "pT_1": [10.0, 20.0],
        "pT_2": [5.0,  7.0],
        "label": [0, 1],
        "weight": [1.0, 2.0],
        "event": [100, 101],
    })

    spec = DatasetSpec(
        features=("pT_1", "pT_2"),
        label="label",
        weight="weight",
        event_id="event",
    )

    arr = build_arrays(df, spec, dtype=np.float64)
    assert arr.X.shape == (2, 2)
    assert arr.y.tolist() == [0, 1]
    assert arr.w is not None and arr.w.dtype == np.float64
    assert "event_id" in arr.meta

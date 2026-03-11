import numpy as np
from eventclf.io import RootReader
from eventclf.config import HMUMU_ZH2L_SCHEMA
from eventclf.data import DatasetSpec, build_arrays
from eventclf.utils.io import save_json
from eventclf.tuning.search_spaces import xgb_search_space_v1
from eventclf.tuning.objective import xgb_cv_objective
from eventclf.tuning.tuner import random_search


def main():
    # Load data
    reader = RootReader(tree_name="Hmumu_SR", step_size="200 MB")
    schema = HMUMU_ZH2L_SCHEMA

    branches = list(schema.feature_names()) + [schema.label, schema.weight, schema.event_id]
    df = reader.read(files=["your.root"], branches=branches)

    spec = DatasetSpec(
        features=schema.feature_names(),
        label=schema.label,
        weight=schema.weight,
        event_id=schema.event_id,
    )
    arr = build_arrays(df, spec, dtype=np.float64)

    fold_id = arr.meta["event_id"] % 4  # your fold logic
    X, y, w = arr.X, arr.y, arr.w

    space = xgb_search_space_v1()

    def obj(params):
        return xgb_cv_objective(
            params,
            X=X, y=y, w=w, fold_id=fold_id,
            metric="auc",
            n_folds=4
        )

    res = random_search(space=space, objective_fn=obj, n_trials=30, seed=123)

    print("Best score:", res.best_score)
    print("Best params:", res.best_params)

    save_json("configs/xgb_hmumu_zh2l_best_v1.json", {
        "best_score": res.best_score,
        "best_params": res.best_params,
        "n_trials": 30,
    })


if __name__ == "__main__":
    main()

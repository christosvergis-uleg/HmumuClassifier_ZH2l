from __future__ import annotations
import os
import random
from typing import Optional

import numpy as np

def set_global_seed(seed: int, set_hash_seed: bool = True) -> None:
    """
    Sets seeds for python, numpy, and (optionally) PYTHONHASHSEED.
    For torch/tf you'd add hooks in your model code, not here.
    """
    random.seed(seed)
    np.random.seed(seed)

    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)
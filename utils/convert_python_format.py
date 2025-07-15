import pandas as pd
import numpy as np


def to_python_format(val):
    if pd.isna(val): return None

    if isinstance(val, np.generic):
        return val.item()  # np.int64, np.float64 → int, float

    return val

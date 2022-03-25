import pandas as pd

def _check_types(s):

    if not isinstance(s, pd.Series):
        raise ValueError("Input should be pandas series.")

    try:
        first_non_nan_value = s.loc[s.first_valid_index()]
        if not isinstance(first_non_nan_value, str):
            raise ValueError("Pandas series should consist of only text.")
    except KeyError:  # Only NaNs in Series -> same warning applies
        raise ValueError("Pandas series should consist of only text.")

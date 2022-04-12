from typing import Tuple
import pandas as pd
import functools


def validate_text_input(s: pd.Series) -> None:
    """
    Validate input to be Pandas Series with Text.
    Raises ValueError if fails
    """
    if not isinstance(s, pd.Series):
        raise ValueError("The input should be pandas series.")

    try:
        first_non_nan_value = s.loc[s.first_valid_index()]
        if not isinstance(first_non_nan_value, str):
            raise ValueError("The input pandas series should only consist of strings.")
    except KeyError:  # Only NaNs in Series -> same warning applies
        raise ValueError("The input pandas series should only consist of strings.")


# Deprecated Input Decorator
def __check_types(s: pd.Series) -> Tuple[bool, str]:
    """
    Check type of input.
    """

    if not isinstance(s, pd.Series):
        return False, "The input should be pandas series."

    try:
        first_non_nan_value = s.loc[s.first_valid_index()]
        if not isinstance(first_non_nan_value, str):
            return False, "The input pandas series should only consist of strings."
    except KeyError:  # Only NaNs in Series -> same warning applies
        return False, "The input pandas series should only consist of strings."

    return True, ""


def TextSeries(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        s = args[0]  # The first input argument will be checked.
        # Check if input series can fulfill type.

        fulfills, error_string = __check_types(s)
        if not fulfills:
            raise TypeError(error_string)

        # If we get here, the type can be fulfilled -> execute function as usual.
        return func(*args, **kwargs)

    return wrapper

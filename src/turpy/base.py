from typing import Any, Union, List, Callable

AnySklearnEstimator = Any
TokenizerFunc = Union[Callable[[str], List[str]], None]

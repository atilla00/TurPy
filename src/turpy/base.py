from typing import Any, Optional, List, Callable

AnySklearnEstimator = Any
TokenizerFunc = Optional[Callable[[str], List[str]]]

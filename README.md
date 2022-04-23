# TurPy

![Logo](TurPyLogo.png)

**TurPy** is a scikit-learn wrapper around multiple NLP techniques and libraries for ease of use. It specifically focuses on Turkish language.

## Installation

For general purpose:

    pip install turpy-nlpkit

If you want to use transformer based models using CPU.

    pip install pytorch
    pip install turpy-nlpkit[deep]

If you want to use transformer based models using GPU it preferred you use conda to install cuda dependencies.

    conda install pytorch>=1.6 cudatoolkit=11.0 -c pytorch
    pip install turpy-nlpkit[deep]

## Examples

```python
from turpy.models import Doc2VecClassifier
import pandas as pd

X = pd.Series(["bunu hiç beğenmedim", "bence konusuyla oyunculuğuyla başarılı bir film", "boş zamanınız varsa izleyen, kült bir yapıt sayılmaz"])
y = pd.Series(["negatif","positif", "notr"])

X_test = pd.Series(["bence başarılı"])

model = Doc2VecClassifier()

model.fit(X, y)
model.predict(X_test)
```

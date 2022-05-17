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

    conda install pytorch cudatoolkit=11.3 -c pytorch
    pip install turpy-nlpkit[deep]

## Quick Start

For classification tasks, you can use classes from turpy.models.

```python
from turpy.models import Doc2VecClassifier
import pandas as pd

X = pd.Series(["bunu hiç beğennmedim", "bence konusuyla oyunculuguyla basarili bir film", "boş zamanınız varsa izleyn, kült bir yapıt sayılmaz"])
y = pd.Series(["negatif","positif", "notr"])

X_test = pd.Series(["bence başarılı"])

model = Doc2VecClassifier()
# or 
# model = Doc2VecClassifier(estimator=any_sklearn_compatible_estimator)


model.fit(X, y)
model.predict(X_test)
```

For preprocessing/spelling_correction/augmentation its better to use sklearn pipeline adding custom processing flow.
In addition, TextPreprocesser may also combine multiple functions in order.

**Note: Importing preprocessing package might take 5-10 seconds because it loads default word dictionary for Turkish.

```python
from turpy.preprocess import TextPreprocesser
from sklearn.pipeline import Pipeline
import pandas as pd

X = pd.Series(["bunu hiç beğennmedim...", "bence konusuyla oyunculuguyla başarı1li bir film", "boş zamanın varsa izle 🙃"])

# Using sklearn pipelines
pipe = Pipeline([
    ('remove_punctuations', TextPreprocesser(replace_punctuations=True)),
    ('replace_digits', TextPreprocesser(replace_digits=True)),
    ('emojis_to_tokens', TextPreprocesser(replace_emojis="<EMOJI>")),
])

# or use like this
# pipe = TextPreprocesser(replace_punctuations=True, replace_digits=True, replace_emojis="<EMOJI>", order=["replace_punctuations", "replace_digits", "replace_emojis"])
# if the apply order doesn't matter (predefined order is used)
# pipe = TextPreprocesser(replace_punctuations=True, replace_digits=True, replace_emojis="<EMOJI>")

X_transformed = pipe.transform(X)

```

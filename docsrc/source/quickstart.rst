Quickstart
===========================


A basic classification example using Doc2VecClassifier

::

    from turpy.models import Doc2VecClassifier
    import pandas as pd

    X = pd.Series(["bunu hiç beğenmedim", "bence konusuyla oyunculuğuyla başarılı bir film", "boş zamanınız varsa izleyen, kült bir yapıt sayılmaz"])
    y = pd.Series(["negatif","positif", "notr"])

    X_test = pd.Series(["bence başarılı"])

    model = Doc2VecClassifier()

    model.fit(X, y)
    model.predict(X_test)
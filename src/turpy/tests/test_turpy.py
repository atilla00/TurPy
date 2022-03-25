from turpy.preprocess import TextPreprocesser
import pandas as pd

def test_answer():
    processor = TextPreprocesser(lowercase=True)
    data = pd.Series(["AAAAA"])
    transformed_data = processor.fit_transform(data)

    data == transformed_data

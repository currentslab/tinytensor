from tinytensor.pipeline import make_pipeline



def test_multi_model():
    model = make_pipeline("models/multi-transformers-epoch-09")
    model.predict({"tokens": "測試這三小"})
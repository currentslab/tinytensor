from tinytensor.pipeline import make_pipeline



def test_multi_model():
    model = make_pipeline("models/multi-transformers-epoch-09")
    output = model.predict({"tokens": "testing this 123"})
    assert isinstance(output, dict)

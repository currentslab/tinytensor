from tinytensor.pipeline import make_pipeline



def test_multi_model():
    model_checkpoints = ["models/multi-transformers-epoch-09",
        "models/multi-gru-epoch-10"]

    for checkpoint in model_checkpoints:
        model = make_pipeline(checkpoint)
        output = model.predict({"tokens": "Apple 3rd quadter earnings had imrpive"})
        assert isinstance(output, dict)
        assert 'emotion_score' in output
        assert 'sentiment_score' in output
        assert 'multi_hierarchical_category' in output

        output = model.predict({"tokens": ["Apple 3rd quadter earnings had imrpive", "I have high probability to assume you had nothing to do"]})
        assert isinstance(output, dict)
        assert 'emotion_score' in output
        assert 'sentiment_score' in output
        assert 'multi_hierarchical_category' in output


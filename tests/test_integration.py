from tinytensor.pipeline import make_pipeline

def test_single_class():

    model = make_pipeline('models/news_categorization')
    output = model.predict({"input_ids": ["Apple 3rd quadter earnings had imrpive", "What to do with this?"]})
    assert len(output['class']) == 1
    assert output['class'][0][0] == 'technology'

def test_multi_model():
    model_checkpoints = [
        "models/multi-transformers-epoch-09",
        "models/multi-transformers-v3",
        "models/multi-gru-epoch-10"
    ]

    for checkpoint in model_checkpoints:
        model = make_pipeline(checkpoint)
        output = model.predict({"tokens": "Apple 3rd quadter earnings had imrpive"})
        assert isinstance(output, dict)
        assert 'emotion_score' in output
        assert 'sentiment_score' in output
        assert 'multi_hierarchical_category' in output
        output = model.predict({"tokens": ["Apple 3rd quadter earnings had imrpive", 
            "I have high probability to assume you had nothing to do", 
            "美商務部長雷蒙多：台海影響 豁免中國貨品關稅難度提升",
            "《黑話律師》「市長」金柱憲特別出演《非常律師禹英禑》大結局！曾和劉仁植導演合作《浪漫醫生金師傅2》 ",
            " 2,999 人民幣有 Snapdragon 8+、1 億像素 OIS 主鏡!Redmi K50 至尊版發佈 "]})
        assert isinstance(output, dict)
        assert 'emotion_score' in output
        assert 'sentiment_score' in output
        assert 'multi_hierarchical_category' in output
        assert isinstance(output['multi_hierarchical_category'], list)
        # print(checkpoint, output['multi_hierarchical_category'])



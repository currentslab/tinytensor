import numpy as np
from tinytensor.postprocessing.classification import HierarchicalMultiClassification
from tinytensor.math_utils import sigmoid
from tinytensor.preprocessing.tokenizer import TextProcessing


def test_hierarchical():
    class_mapping = 'models/multi-transformers-epoch-09/class2idx.json'
    configuration = {
        "mapping": "class2idx.json",
        'level_class': [27, 644],
        '_dir': 'models/multi-transformers-90k',
        'threshold': 0.5
    }
    postprocessing = HierarchicalMultiClassification(configuration)


    logits = -np.ones((32, 644))
    logits[:3, :100] =  np.random.random((3, 100))
    logits[:3, :10] = np.ones((3, 10))
    output = postprocessing(sigmoid(logits), {})
    for zero_output in output[3:]:
        assert len(zero_output) == 0


def test_tokenizer():
    preprocessing = TextProcessing({
        '_dir': 'models/multi-transformers-90k',
        'tokenizer_file': 'tokenizer-90k.json',
        'name': 'tokens',
        'pad_token': '[PAD]'
    })
    inputs = {"tokens": ["Apple 3rd quadter earnings had imrpive", "I have high probability to assume you had nothing to do"]}
    output = preprocessing(inputs, {})
    assert len(output['tokens'].shape) > 1
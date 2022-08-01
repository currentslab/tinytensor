import numpy as np
from tinytensor.postprocessing.classification import HierarchicalMultiClassification
from tinytensor.math_utils import sigmoid



def test_hierarchical():
    class_mapping = 'models/multi-transformers-epoch-09/class2idx.json'
    configuration = {
        'level_class': [27, 644],
        'threshold': 0.5
    }
    postprocessing = HierarchicalMultiClassification(class_mapping, configuration)


    logits = -np.ones((32, 644))
    logits[:3, :100] =  np.random.random((3, 100))
    logits[:3, :10] = np.ones((3, 10))
    outputs = {
        'hierarchical': sigmoid(logits)
    }
    output = postprocessing.forward(outputs, {})
    for zero_output in output[3:]:
        assert len(zero_output) == 0


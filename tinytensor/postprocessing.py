import logging
from tinytensor.postprocessing.text_classification import (
    HierarchicalMultiClassification,
    Classification,
    TopkClassification
)

pipeline2class = {
    'topk_classification': TopkClassification,
    'classification': Classification,
    'multi_hierarchical_classification': HierarchicalMultiClassification
}

def make_postprocessing(main_configuration):
    processor = []
    
    pipeline = [ (int(idx), preprocess_config) for idx, preprocess_config in main_configuration['outputs']]
    pipeline = sorted(pipeline)
    for (order_id, config) in pipeline:
        type_ = config['output']
        potsprocess_cls = pipeline2class[type_]
        processor.append(
            potsprocess_cls(config)
        )
    
    return processor



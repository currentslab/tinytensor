import logging
import json
import numpy as np
from .abstract import Postprocessing
from tinytensor.math_utils import sigmoid

class HierarchicalMultiClassification(Postprocessing):
    
    def __init__(self, class_mapping: str, configuration) -> None:
        super().__init__()
        with open(class_mapping, 'r') as f:
            mapping = json.load(f)        
        self.config = configuration
        self.mapping2idx = mapping
        self.idx2mapping = { idx: cls_str for cls_str, idx in mapping.items() }
        self.levels = self.config['level_class']
        self.threshold = self.config['threshold']

    def forward(self, outputs, chains):
        logits = outputs['hierarchical']

        batch_s, cls_s = logits.shape
        # array of batch_size x sum(self.levels)
        prev_level = 0

        items = [ {} for _ in range(batch_s)]
        for idx, logit in enumerate(logits):
            prev_level = 0
            logit_cls = np.argwhere(logit > self.threshold).flatten()

            for level in self.levels: # iterate through all levels
                classes = logit_cls[(logit_cls < level) & (logit_cls > prev_level)]
                if len(classes) == 0:
                    break

                if prev_level == 0:
                    items[idx] = { self.idx2mapping[cls]: logits[idx][cls] \
                            for cls in classes }
                elif len(items[idx]) > 0:
                    for cls in classes:
                        pred_map = self.idx2mapping[cls]
                        # hierarchical loss only add if previous level exists
                        if pred_map.split('/')[0] in items[idx]:
                            items[idx][pred_map] = logits[idx][cls]

                prev_level = level
        # aggregate results later
        return items



class Classification(Postprocessing):

    def __init__(self, class_mapping: str, configuration) -> None:
        super().__init__()
        with open(class_mapping, 'r') as f:
            mapping = json.load(f)        
        self.config = configuration
        self.idx2mapping = { idx: cls_str for cls_str, idx in mapping.items() }


    def forward(self, outputs, chains):
        logits = outputs['classification']
        assert len(logits.shape) == 2
        outputs = []
        for prob in logits:
            cls = np.argmax(prob)
            outputs.append(( self.idx2mapping[cls], prob[cls] ))
        return outputs
    

class TopkClassification(Postprocessing):
    def __init__(self, class_mapping: str, configuration) -> None:
        super().__init__()
        with open(class_mapping, 'r') as f:
            mapping = json.load(f)        
        self.config = configuration
        self.idx2mapping = { idx: cls_str for cls_str, idx in mapping.items() }
        self.K = self.config['top_k']

    def forward(self, outputs, chains):
        logits = outputs['classification']
        assert len(logits.shape) == 2
        outputs = []
        for prob in logits:
            indices = np.argpartition(prob, -self.K)[-self.K:]
            top_scores = []
            for index in indices:
                top_scores.append((self.idx2mapping[index], prob[index]))
            top_scores.sort(reverse=True)
            outputs.append(top_scores)
        return outputs


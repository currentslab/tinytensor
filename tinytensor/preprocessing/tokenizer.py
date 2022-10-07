import logging
import os
import numpy as np
try:
    from tokenizers import Tokenizer
except ImportError:
    logging.warning("Huggingface tokenizers not installed ")
from tinytensor.preprocessing.abstract import Preprocessing


class TextProcessing(Preprocessing):

    def __init__(self, configuration) -> None:
        self.tokenizer = Tokenizer.from_file(
            os.path.join(configuration['_dir'], configuration['tokenizer_file'])
        )
        self.name = configuration["name"]
        self.return_attention = False
        if 'return_attention' in configuration:
            self.return_attention = configuration['return_attention']
        self.pad_token = self.tokenizer.token_to_id(configuration["pad_token"])
        if self.pad_token is None:
            self.pad_token = 0

    def __call__(self, inputs, model_inputs):
        texts = inputs[self.name]
        if isinstance(texts, str):
            texts = [texts]

        output = []
        max_length = 0
        for text in texts:
            encode = self.tokenizer.encode(text)
            output.append(encode.ids)
            max_length = max(len(encode.ids), max_length)
        np_output = []
        attn_output = []
        for token in output:
            t_len = len(token)
            pad_len = max(max_length-t_len, 0)
            if pad_len > 0:
                token += [self.pad_token]*pad_len
            np_output.append(token)
            if self.return_attention:
                attn_output.append([1]*t_len + [0]*pad_len)


        if self.return_attention:
            model_inputs['attention_mask'] = np.array(attn_output)
        model_inputs[self.name] = np.array(np_output)
        return model_inputs


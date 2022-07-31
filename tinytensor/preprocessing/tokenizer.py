import logging
try:
    import tokenizer
except ImportError:
    logging.warning("Huggingface tokenizers not installed ")

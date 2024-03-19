"""
TinyChat tokenizer and tokenization utilities
"""

from typing import NoReturn as NoReturn

from .TinyChatTokenizer import Tokenizer
from .TokenizerUtils import set_default_tokenizer, count_tokens, check_special_tokens, train_multiple_corpus, optimize_multiple_corpus, get_encoding_vocab_multiple_corpus


__all__ = [
    "Tokenizer",
    "set_default_tokenizer",
    "count_tokens",
    "check_special_tokens",
    "train_multiple_corpus",
    "optimize_multiple_corpus",
    "get_encoding_vocab_multiple_corpus"
]
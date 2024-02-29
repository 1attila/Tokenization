"""
TinyChat tokenizer and tokenization utilities
"""

from typing import NoReturn as NoReturn

from .TinyChatTokenizer import Tokenizer
from .TokenizerUtils import count_tokens, check_special_tokens, default_tokenizer


__all__ = [
    "Tokenizer",
    "count_tokens",
    "check_special_tokens",
    "set_default_tokenizer"
]


def set_default_tokenizer(tokenizer: Tokenizer) -> NoReturn:
    """
    Sets the default tokenizer
    
    Args:
      `tokenizer` (TinyChatTokenizer): Default tokenizer to ues

    Returns:
      NoReturn
    """

    global default_tokenizer
    default_tokenizer = tokenizer
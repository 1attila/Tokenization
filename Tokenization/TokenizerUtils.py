from typing import Optional
import regex as re

from .TinyChatTokenizer import Tokenizer


default_tokenizer: Tokenizer | None = None


def get_tokenizer(tokenizer: Optional[Tokenizer]):
    return default_tokenizer if tokenizer is None else tokenizer


def count_tokens(text: str, tokenizer: Optional[Tokenizer], include_special_tokens: bool=True) -> int:
    """
    Counts the number of tokens present in the given string

    Args:
      `text` (str): Text to encode
      `tokenizer` (Tokenizer, optional): Tokenizer to use. `default_tokenizer` by default
      `include_special_tokens` (bool): Count special tokens or not. True by default

    Returns:
      int: Number of tokens in the string

    Raises:
      AssertionError: If text is not a string
      AssertionError: If the string is empty
      ValueError: If unknown tokens are found
    """

    assert isinstance(text, str), "text must be a string!"
    assert len(text) > 0, "text can't be empty!"

    tokenizer = get_tokenizer(tokenizer)

    if include_special_tokens:
        return len(tokenizer.encode(text))
    else:
        chunks = [text]
        chunks = re.split(tokenizer.SpecialTokensPattner, text)

        ids = []

        for chunk in chunks:

            if chunk not in tokenizer.SpecialTokens:
                ids.extend(tokenizer.Vocab[chunk])

        return len(ids)


def check_special_tokens(text: str, tokenizer: Optional[Tokenizer]) -> bool:
    """
    Check if special tokens are in the prompt

    Args:
      `text` (str): Text to check
      `tokenizer` (Tokenizer. optional): Tokenizer to use. `default_tokenizer` by default

    Returns:
      bool: True if the text contains special tokens or not

    Raises:
      AssertionError: If text is not a string
      AssertionError: If text is empty
    """

    assert isinstance(text, str), "text must be a string!"
    assert len(text) > 0, "text can't be empty!"

    tokenizer = get_tokenizer(tokenizer)

    return len(re.split(tokenizer.SpecialTokensPattner, text)) > 0
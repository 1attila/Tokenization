from typing import Optional, NoReturn, Union
import regex as re

from .TinyChatTokenizer import Tokenizer
from .utils import read_corpus


default_tokenizer: Tokenizer | None = None


def set_default_tokenizer(tokenizer: Tokenizer) -> NoReturn:
    """
    Sets the default tokenizer
    
    Args:
      `tokenizer` (TinyChatTokenizer): Default tokenizer to use

    Returns:
      NoReturn
    """

    global default_tokenizer
    default_tokenizer = tokenizer


def get_tokenizer(tokenizer: Optional[Tokenizer]):
    return default_tokenizer if tokenizer is None else tokenizer


def count_tokens(text: str, tokenizer: Optional[Tokenizer]=None, include_special_tokens: bool=True) -> int:
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


def check_special_tokens(text: str, tokenizer: Optional[Tokenizer]=None) -> bool:
    """
    Check if special tokens are in the prompt

    Args:
      `text` (str): Text to check
      `tokenizer` (Tokenizer, optional): Tokenizer to use. `default_tokenizer` by default

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


def train_multiple_corpus(corpus_paths: Union[list[str], str], max_vocab_size: int, tokenizer: Optional[Tokenizer]=None, model_name: str="TinyChat-1") -> NoReturn:
    """
    Trains the tokenizer on the given corpus

    Args:
      -`corpus_paths` (list | str): Paths of the corpus. Could be a folder or list of files
      -`tokenizer` (Tokenizer, optional): Tokenizer to use. `default_tokenizer` by default
      -`max_vocab_size` (int): The number of vocabs. NOTE: It must be greather than 256!
      -`model_name` (str, optional): Name of the .model file to save the tokenizer. "TinyChat-1" by default

    Returns:
      -NoReturn

    Raises:
      -AssertionError: If max_vocab_size is less or equal than 256
    """
    
    assert max_vocab_size > 256, "max_vocab_size should be greater than 256!"

    tokenizer = get_tokenizer(tokenizer)

    total_corpus: str = ""

    for corpus in read_corpus(corpus_paths):
        total_corpus += corpus
    
    tokenizer.train(text=total_corpus, vocab_size=max_vocab_size, reset=True, model_name=model_name)


def optimize_multiple_corpus(corpus_paths: Union[list[str], str], tokenizer: Optional[Tokenizer]=None, include_special_tokens: bool=False, model_name: str="TinyChat-1") -> NoReturn:
    """
    Optimizes the tokenizer vocab trained on multiple corpus

    Args:
      -`corpus_paths` (list | str): Paths of the corpus. Could be a folder or a list of files
      -`tokenizer` (Tokenizer, optional): Tokenier to use. `default_tokenizer` by default
      -`include_special_tokens` (bool, optional): If true, it will remove unused special tokens
      -`model_name` (str, optional): Name of the .model file to save the tokenizer. "TinyChat-1" by default
    
    Returns:
      -NoReturn
    """

    tokenizer = get_tokenizer(tokenizer)

    tokens_used: set[int] = set()

    for corpus in read_corpus(corpus_paths):
        tokens_used = tokens_used | tokenizer.get_model_vocab(corpus)

    for item in dict(tokenizer.Merges.keys()):
        
        to_remove: bool = True

        for a, b in tokenizer.Merges.keys():
                
            if item == a or item == b or item in tokens_used:
                to_remove = False

        if to_remove:
            del tokenizer.Vocab[item]

    if include_special_tokens:

        for item in dict(tokenizer.SpecialTokens).values():

            if item not in tokens_used:
                tokenizer.SpecialTokens.pop(tokenizer.SpecialTokensReverse[item])
                tokenizer.SpecialTokensReverse.pop(item)
    
    tokenizer.save(model_name)


def get_encoding_vocab_multiple_corpus(corpus_paths: Union[list[str], str], tokenizer: Optional[Tokenizer]=None) -> NoReturn:
    """
    Builds the mapped vocabs to communicate with the LLM

    Args:
      -`corpus_paths` (list | str): Paths of the corpus. Could be a folder or a list of files
      -`tokenizer` (Tokenizer, optional): Tokenier to use. `default_tokenizer` by default

    Returns:
      -NoReturn
    """

    tokenizer = get_tokenizer(tokenizer)

    total_corpus: str = ""

    for corpus in read_corpus(corpus_paths):
        total_corpus += corpus
        
    tokenizer.get_decoding_vocab(total_corpus)
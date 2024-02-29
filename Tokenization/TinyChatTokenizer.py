import regex as re
from typing import NoReturn, Optional


PATTNER = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    TinyChat tokenizer based on utf-8 encoding
    Current version: 1.0

    Methods:
      -`__init__(merges: Optional[dict[tuple[int, int], int]]=None, special_tokens: Optional[dict[str, int]]=None) -> NoReturn`
      -`from_folder(model_name: str="TinyChat-1") -> NoReturn`
      -`train(text: str, vocab_size: int, reset: bool, model_name: str="TinyChat-1") -> Tokenizer`
      -`encode_normal(text: str) -> list[int]`
      -`encode(text: str) -> list[int]`
      -`decode(tokens: list[int]) -> str`
      -`get_model_vocab(text: str) -> set[int]`
    """

    Pattner: re.Pattern = re.compile(PATTNER)
    Vocab: dict[int, bytes] = {}
    Merges: dict[tuple[int, int], int] = {}
    SpecialTokens: dict[str, int] = {}
    SpecialTokensReverse: dict[int, str] = {}
    SpecialTokensPattner: str = ""
    VERSION: str = "1.0"


    def __init__(self, merges: Optional[dict[tuple[int, int], int]]=None, special_tokens: Optional[dict[str, int]]=None) -> NoReturn:
        """
        Builds the tokenizer with a specific merges and special tokens dict

        Args:
          `merges` (dict, optional): All possibles bytes merging. Example: given {(0, 1): 2} will encode [0, 1] into [2]\n
          `special_tokens` (dict, optional): All the special tokens. None by default. NOTE: Special tokens ids must to be greather than vocab_size. Example: {"eos": 1001} (vocab_size < 1000)

        Returns:
          NoReturn

        Raises:
          AssertionError: If merges is not a dict
          AssertionError: If special_tokens is not a dict
          AssertionError: If a special token id is less than vocab_size
        """

        merges = {} if merges is None else merges

        assert isinstance(merges, dict), "merges must be a dict!"

        self.Merges = merges

        self.Vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in merges.items():
            self.Vocab[idx] = bytes(self.Vocab[p0] + self.Vocab[p1])

        if special_tokens is not None:

            assert isinstance(special_tokens, dict), "special_tokens must be a dict!"

            self.SpecialTokens = special_tokens
            self.SpecialTokensReverse = {v: k for k, v in special_tokens.items()}

            for item in special_tokens.values():
                assert item > len(self.Merges) + 256, f"The {item} id for special token needs to be grather than {len(self.Merges) + 256}!"

            self.SpecialTokensPattner = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"


    def get_stats(self, ids: list[int], counts: Optional[dict[tuple[int, int], int]]=None) -> dict[tuple[int, int], int]:

        counts: dict = {} if counts is None else counts

        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts
    

    def replace_pair(self, ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:

        new_ids: int = []
        i: int = 0

        while i < len(ids):

            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids


    def train(self, text: str, vocab_size: int, reset: bool=False, model_name: str="TinyChat-1") -> NoReturn:
        """
        Trains the tokenizer on the given corpus

        Args:
          `text` (str): Text to train the tokenizer
          `vocab_size` (int): Lenght of vocab size. NOTE: It must be greater than 256!
          `reset` (bool, optional): If set to `True` it will not reset the precedent merges from other trainings. False by default
          `model_name` (str): Name of .model file to save the tokenizer. "TinyChat-1" by default

        Returns:
          NoReturn

        Raises:
          AssertionError: If text is not a string
          AssertionError: If text is empty
          AssertionError: If vocab_size is less than 256
        """
        
        assert isinstance(text, str), "text must be a string!"

        text = re.sub(self.SpecialTokensPattner, "", text)

        assert len(text) != 0, "text must contain at least a character (special tokens excluded)!"

        assert vocab_size > 256, "vocab_size should be greater than 256!"

        merges = self.Merges
        vocab = self.Vocab

        num_merges: int = vocab_size - 256
        merge_offset: int = len(merges)

        if reset:
            merges = {}
            vocab = {idx: bytes([idx]) for idx in range(256)}
            merge_offset = 256

        text_chunks = re.findall(self.Pattner, text)

        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_merges - merge_offset):
            stats = {}
            i += merge_offset

            for chunks_ids in ids:
                self.get_stats(chunks_ids, stats)
            
            pair = max(stats, key=stats.get, default=0)
            idx = i
            
            ids = [self.replace_pair(chunk_ids, pair, idx) for chunk_ids in ids]
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.Merges = merges
        self.Vocab = vocab

        with open(f"{model_name}.model", "w", encoding="utf-8") as f:

            f.write(f"Version: {self.VERSION} \n")
            f.write("Merges: \n")
            
            for idx1, idx2 in self.Merges:
                f.write(f"{idx1} {idx2} \n")

            if len(self.SpecialTokens) > 0:

                f.write("SpecialTokens: \n")
                for token, id in self.SpecialTokens.items():
                    f.write(f"{token} {id} \n")

    
    @staticmethod
    def from_folder(model_name: str="TinyChat-1") -> "Tokenizer":
        """
        Builds the tokenizer from a directory

        Args:
          `model_name` (str, optional): Name of .model file containing the tokenizer attributes. "TinyChat-1" by default

        Returns:
          Tokenizer: Tokenizer configured with the given file attributes

        Raises:
          AssertionError: If file version and tokenizer version doesn't match
          AssertionError: If file was written incorrectly
        """
        
        with open(f"{model_name}.model", "r", encoding="utf-8") as f:

            merges: dict[tuple[int, int], int] = {}
            special_tokens: dict[str, int] = {}
            idx: int = 256
            contains_special_tokens: bool = False

            with open(f"{model_name}.model", "rb") as f2:
                file_lenght = sum(1 for _ in f2)
            
            assert f.readline() == f"Version: {Tokenizer.VERSION} \n", f"The file version and the tokenizer version doesn't match, it should be {Tokenizer.VERSION}"
            assert f.readline() == "Merges: \n", "File doesn't match the grammar!"
            
            for _ in range(file_lenght):

                line = f.readline()

                if line == "SpecialTokens: \n":
                    contains_special_tokens = True
                    break

                else:
                    idx1, idx2 = map(int, line.split())
                    merges[(idx1, idx2)] = idx
                    idx += 1

            if contains_special_tokens:

                for _ in range(file_lenght):

                    line = f.readline()

                    if len(line) > 0:
                        token, id = map(str, line.split())
                        special_tokens[token] = int(id)
                    else:
                        break

            vocab: dict = {idx: bytes([idx]) for idx in range(256)}

            for (p0, p1), idx in merges.items():
                vocab[idx] = bytes(vocab[p0] + vocab[p1])

            return Tokenizer(merges, special_tokens)
        
    
    def encode_chunk(self, text_bytes: list[bytes]) -> list[int]:

        ids = list(text_bytes)

        while len(ids) >= 2:

            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.Merges.get(p, float("inf")))
            
            if pair not in self.Merges:
                break

            idx = self.Merges[pair]
            ids = self.replace_pair(ids, pair, idx)

        return ids
    

    def encode_normal(self, text: str) -> list[int]:
        """
        Encodes the given text into tokens
        NOTE: this functions doesn't accept special tokens!

        Args:
          `text` (str): String to encode

        Returns:
          list: Encoded tokens

        Raises:
          AssertionError: If the text is not a string
          ValueError: If unknown tokens are found
        """
        
        assert isinstance(text, str), "`text` must be a string!"

        ids: list[int] = []

        text_chunks = re.findall(self.Pattner, text)

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_idx = self.encode_chunk(chunk_bytes)
            ids.extend(chunk_idx)

        return ids
    

    def encode(self, text: str) -> list[int]:
        """
        Encodes the given text into tokens

        Args:
          `text` (str): String to encode

        Returns:
          list: Encoded tokens

        Raises:
          AssertionError: If text is not a string
          ValueError: If unknown tokens are found
        """

        assert isinstance(text, str), "`text` must be a string!"

        ids: list[int] = []

        special_chunks = [text]

        if len(self.SpecialTokens) > 0:
            special_chunks = re.split(self.SpecialTokensPattner, text)
        
        for part in special_chunks:
            
            if part in self.SpecialTokens:
                ids.append(self.SpecialTokens[part])
            else:
                ids.extend(self.encode_normal(part))

        return ids


    def decode(self, tokens: list[int]) -> str:
        """
        Decodes the given tokens into text

        Args:
          `tokens` (list): List of tokens to decode

        Returns:
          str: Decoded tokens

        Raises:
          AssertionError: If tokens is not a list
          ValueError: If unknown tokens are found
        """

        assert isinstance(tokens, list), "`tokens` must be a list of integers!"

        decoded_bytes: list[bytes] = []
        
        for item in tokens:

            if item in self.Vocab:
                decoded_bytes.append(self.Vocab[item])

            elif item in self.SpecialTokensReverse:
                decoded_bytes.append(self.SpecialTokensReverse[item].encode("utf-8"))
        
            else:
                raise ValueError(f"Invalid token id: `{item}`")
        
        decoded_bytes = b"".join(decoded_bytes)
        decoded_text = decoded_bytes.decode("utf-8", errors="replace")

        return decoded_text

    
    def get_model_vocab(self, text: str) -> set[int]:
        """
        Returns the set of tokens used for encoding the given text

        Args:
          `text` (str): Text to encode

        Returns:
          set: Vocab of tokens used to encode the given text

        Raises:
          AssertionError: If text is not a string
          AssertionError: If text is empty
        """

        assert isinstance(text, str), "`text` must be a string!"
        assert len(text) > 0, "`text` must contain at least a carachter!"

        return set(self.encode(text))
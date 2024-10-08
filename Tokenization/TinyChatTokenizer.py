import regex as re
from typing import NoReturn, Optional


PATTNER = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    TinyChat tokenizer based on utf-8 encoding
    Current version: 1.0

    Methods:
      -`__init__(merges: Optional[dict[tuple[int, int], int]]=None, vocab:Optional[int, bytes],special_tokens: Optional[dict[str, int]]=None) -> NoReturn`
      -`from_folder(model_name: str="TinyChat-1") -> NoReturn`
      -`save(model_name: str="TinyChat-1")`
      -`train(text: str, vocab_size: int, reset: bool, model_name: str="TinyChat-1") -> Tokenizer`
      -`optimize(text: str, include_special_tokens: bool=False, model_name: str="TinyChat-1") -> NoReturn`
      -`get_decoding_vocab(text: str) -> NoReturn`
      -`apply_encoding_vocab(tokenized_text: list[int]) -> list[int]`
      -`apply_decoding_vocab(tokens: list[int]) -> list[int]`
      -`encode_normal(text: str) -> list[int]`
      -`encode(text: str) -> list[int]`
      -`decode(tokens: list[int]) -> str`
      -`get_model_vocab(text: str) -> set[int]`
    """

    Pattner: re.Pattern = re.compile(PATTNER)
    Vocab: dict[int, bytes] = {}
    DecodingVocab: dict[int, int] | None = None
    EncodingVocab: dict[int, int] | None = None
    Merges: dict[tuple[int, int], int] = {}
    SpecialTokens: dict[str, int] = {}
    SpecialTokensReverse: dict[int, str] = {}
    SpecialTokensPattner: str = ""
    VERSION: str = "1.0"


    def __init__(self, merges: Optional[dict[tuple[int, int], int]]=None, vocab: Optional[dict[int, bytes]]=None, llm_vocab: Optional[dict[int, int]]=None, special_tokens: Optional[dict[str, int]]=None) -> NoReturn:
        """
        Builds the tokenizer with a specific merges and special tokens dict

        Args:
          `merges` (dict, optional): All possibles bytes merging. Example: given {(0, 1): 2} will encode [0, 1] into [2]
          `vocab` (dict, optional): Vocab to use. Uses `{x: bytes(x) for x in range(256)}` by default
          `special_tokens` (dict, optional): All the special tokens. None by default. NOTE: Special tokens ids must to be greather than vocab_size. Example: {"eos": 1001} (vocab_size < 1000)

        Returns:
          NoReturn

        Raises:
          AssertionError: If merges is not a dict
          AssertionError: If llm_vocab is not a dict
          AssertionError: If special_tokens is not a dict
          AssertionError: If a special token id is less than vocab_size
        """

        merges = {} if merges is None else merges

        assert isinstance(merges, dict), "merges must be a dict!"

        self.Merges = merges

        self.Vocab: dict[int, bytes] = vocab if vocab is not None else {idx: bytes([idx]) for idx in range(256)}

        if len(self.Vocab) <= 256:
          for (p0, p1), idx in merges.items():
              self.Vocab[idx] = bytes(self.Vocab[p0] + self.Vocab[p1])
        
        if llm_vocab is not None:
            assert isinstance(llm_vocab, dict), "llm_vocab must be a dict!"
            self.DecodingVocab = llm_vocab
            self.EncodingVocab = {v: k for k, v in llm_vocab.items()}

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
            vocab: dict[int, bytes] = {}
            special_tokens: dict[str, int] = {}
            contains_special_tokens: bool = False
            llm_vocab: dict[int, int] | None = {}

            with open(f"{model_name}.model", "rb") as f2:
                file_lenght = sum(1 for _ in f2)
            
            assert f.readline() == f"Version: {Tokenizer.VERSION} \n", f"The file version and the tokenizer version doesn't match, it should be {Tokenizer.VERSION}"
            assert f.readline() == "Vocab: \n", "File doesn't match the grammar!"
            
            for _ in range(file_lenght):

                line = f.readline()

                if line == "Merges: \n":
                    break

                else:
                    idx = int(line.strip())
                    vocab[idx] = bytes([idx])
            
            for _ in range(file_lenght):
                
                line = f.readline()

                if line == "SpecialTokens: \n":
                    contains_special_tokens = True
                    break
                
                else:
                    idx1, idx2, idx = map(int, line.split())
                    merges[(idx1, idx2)] = idx

            if contains_special_tokens:

                for _ in range(file_lenght):

                    line = f.readline()

                    if len(line) > 0:
                        token, id = map(str, line.split())
                        special_tokens[token] = int(id)
                    else:
                        break
            
            try:
                with open(f"{model_name}.llmvc", "r", encoding="utf-8") as f:
                
                    for i, line in enumerate(f):
                        
                        if len(line) > 0:
                            llm_vocab[i] = int(line.strip())
                        else:
                            break
            except:
                llm_vocab = None
            
            return Tokenizer(merges=merges, vocab=vocab, llm_vocab=llm_vocab, special_tokens=special_tokens)


    def save(self, model_name: str="TinyChat-1") -> NoReturn:
        """
        Saves the tokenizer merges into a .model file

        Args:
          -`model_name` (str, optional): Name of the .model file to save the tokenizer. "TinyChat-1" by default

        Returns:
          -NoReturn

        Raises:
          -AssertionError: If model_name is empty
        """
        
        assert len(model_name) > 0, "model_name can't be empty!"

        with open(f"{model_name}.model", "w", encoding="utf-8") as f:

            f.write(f"Version: {self.VERSION} \n")

            f.write("Vocab: \n")

            for item in range(256):
                
                if item in self.Vocab.keys() and item < 256:
                  f.write(str(item) + " \n")

            f.write("Merges: \n")
            
            for (idx1, idx2), idx in self.Merges.items():
                f.write(f"{idx1} {idx2} {idx} \n")

            if len(self.SpecialTokens) > 0:

                f.write("SpecialTokens: \n")
                for token, id in self.SpecialTokens.items():
                    f.write(f"{token} {id} \n")

        if self.DecodingVocab is not None:
            with open(f"{model_name}.llmvc", "w", encoding="utf-8") as f:
                for i in self.DecodingVocab.values():
                    f.write(str(i) + " \n")


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
            merge_offset = 0
            vocab = {idx: bytes([idx]) for idx in range(256)}

        text_chunks = re.findall(self.Pattner, text)

        ids = [self.encode_chunk(list(ch.encode("utf-8"))) for ch in text_chunks]

        for idx in range(merge_offset + 256, num_merges + 256):
            
            stats = {}
            for chunks_ids in ids:
                self.get_stats(chunks_ids, stats)
            
            pair = max(stats, key=stats.get)
            
            ids = [self.replace_pair(chunk_ids, pair, idx) for chunk_ids in ids]
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.Merges = merges
        self.Vocab = vocab

        self.save(model_name)


    def optimize(self, text: str, include_special_tokens: bool=False, model_name="TinyChat-1") -> NoReturn:
        """
        Optimizes the tokenizer vocab by removing useless items

        Args:
          -`text` (str): Text to encode
          -`include_special_tokens` (bool, optional): Removes unused tokens if set to true. False by default
          -`model_name` (str, optional): Name of the .model file to save the tokenizer. "TinyChat-1" by default

        Returns:
          -NoReturn

        Raises:
          -AssertionError: If text is not a string
        """

        assert isinstance(text, str), "text must be a string!"

        tokens_used = list(self.get_model_vocab(text))

        #removing all vocabs that aren't in merges and tokens_used
        for item in dict(self.Vocab).keys():
            
            to_remove: bool = True

            for a, b in self.Merges.keys():
                
                if item == a or item == b or item in tokens_used:
                    to_remove = False

            if to_remove:
                del self.Vocab[item]

        if include_special_tokens:

            for item in dict(self.SpecialTokens).values():

                if item not in tokens_used:
                    self.SpecialTokens.pop(self.SpecialTokensReverse[item])
                    self.SpecialTokensReverse.pop(item)

        self.save(model_name)

      
    def get_decoding_vocab(self, text: str) -> NoReturn:
        """
        Builds the mapped vocabs to communicate with the LLM

        Args:
          -`text`(str): Corpus test

        Returns:
          -NoReturn
        """

        used_tokens: set = self.get_model_vocab(text)

        self.DecodingVocab = {k: v for k, v in enumerate(used_tokens)}
        self.EncodingVocab = {v: k for k, v in self.DecodingVocab.items()}


    def apply_encoding_vocab(self, tokenized_text: list[int]) -> list[int]:
        
        """
        Use when you want to send encoded text to LLM

        Args:
          -`tokenized_text`(list): Text encoded by the tokenizer

        Returns:
          -list: List of mapped tokens to send to LLM

        Raises:
          -AssertionError: If EncodingVocab is not set
        """
        #EncodingVocab: dict[int, int] #Tokenizer_tok -> LLM_tok

        assert self.EncodingVocab is not None, "EncodingVocab can't be None"

        #Tokenizer_tokens -> LLM_tokens
        return [self.EncodingVocab[t] for t in tokenized_text]


    def apply_decoding_vocab(self, tokens: list[int]) -> list[int]:

        """
        Use when you want to decode LLM output

        Args:
          -`tokens`(list): Tokens from LLM you want to decode

        Returns:
          -list: Tokens ready to be decoded

        Raises:
          -AssertionError: If DecodingVocab is not set
        """
        #DecodingVocab: dict[int, int] #LLM_tok -> Tokenizer_tok 

        assert self.DecodingVocab is not None, "DecodingVocab can't be None"

        #LLM_tokens -> Tokenizer_tokens
        return [self.DecodingVocab[t] for t in tokens]
        
    
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
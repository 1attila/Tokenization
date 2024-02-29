# Tokenization
A small package with bpe-tokenization utilities for LLMs inspired by Andrej Karpaty's [minbpe][0] repositoy.

## Utilities
### Regex tokenizer
- GPT4 split pattner
- Encoding/decoding methods with custom special tokens dict
- Trainable, even from a precedent tokenizer (useful if you want to train on multiple corpus)
- Supports save/loading from folder
### Utilities
- Count number of tokens
- Check if special tokens are present



## Todo
- Rewrite everything in C++ to make it faster

[0]: https://github.com/karpathy/minbpe

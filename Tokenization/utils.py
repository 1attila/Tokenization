import os
from pathlib import Path
from typing import Iterator


def read_corpus(input: str | list) -> Iterator[str]:
    """
    Yields the corpus present in the given folder/file/text

    Args:
      -`input(str | list)`: Could be a folder_path or file_path

    Returns:
      -Iterator[str]

    Raises:
      -FileNotFoundError: If the specified dir/file does not exist
      -TypeError: If input is neither a list nor a string
    """
    
    assert len(input) > 0, "input can't be empty!"

    if isinstance(input, str):
        
        if os.path.isdir(input):

            for file in os.listdir(input):
                with open(Path(file), "r", encoding="utf-8") as f:
                    yield f.read()
        else:
            raise FileNotFoundError

    elif isinstance(input, list):

        for file in input:
            
            path: Path = Path(file)

            if path.exists:
                with open(Path(file), "r", encoding="utf-8") as f:
                    yield f.read()

            else:
                raise FileNotFoundError

    else:
        raise TypeError
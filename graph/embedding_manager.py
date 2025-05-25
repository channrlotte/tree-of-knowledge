import fasttext

import numpy as np
from numpy.typing import NDArray

ft = fasttext.load_model("embeddings/cc.ru.100.bin")


def get_embedding(word: str) -> NDArray[np.float64]:
    """
    Get the embedding for a given word.

    Args:
        word: The word to get an embedding for.

    Returns:
        A numpy array representing the word embedding.
    """

    return ft.get_word_vector(word)

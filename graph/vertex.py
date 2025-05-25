import numpy as np

from dataclasses import dataclass, field
from typing import Set

from graph.embedding_manager import get_embedding


@dataclass
class Vertex:
    """
    Represents a vertex in the graph representing a concept.

    Attributes:
        concept: The concept this vertex represents.
        embedding: The embedding of the concept.
        adjacent_edges: Set of adjacent edge IDs.
    """

    concept: str
    embedding: np.ndarray = field(init=False)
    adjacent_edges: Set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.embedding = get_embedding(self.concept)

    def __repr__(self) -> str:
        return f"Vertex(concept='{self.concept}')"

    def __hash__(self):
        return hash(self.concept)

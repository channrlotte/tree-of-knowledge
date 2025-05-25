import numpy as np

from dataclasses import dataclass, field
from typing import Set

from graph.embedding_manager import get_embedding


@dataclass
class UnionEdge:
    """
    Represents a group of related edges that form a logical union.

    Attributes:
        edge_ids: Set of IDs of the edges that are part of this union.
        label: The union meaning.
        parent_subgraph: ID of the parent subgraph.
        embedding: Embedding of the meaning.
    """

    edge_ids: Set[int]
    label: str
    parent_subgraph: int
    embedding: np.ndarray = field(init=False)

    def __post_init__(self):
        self.embedding = get_embedding(self.label)

    def __hash__(self) -> int:
        return hash((self.edge_ids, self.label, self.parent_subgraph, self.embedding))

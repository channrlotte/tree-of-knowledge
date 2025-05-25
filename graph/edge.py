import numpy as np

from dataclasses import dataclass, field

from graph.embedding_manager import get_embedding


@dataclass
class Edge:
    """
    Represents an edge in the graph connecting two concepts.

    Attributes:
        agent_1: Source vertex concept.
        agent_2: Target vertex concept.
        label: The relationship meaning.
        parent_subgraph: ID of the parent text.
        embedding: Embedding of the meaning.
    """

    agent_1: str
    agent_2: str
    label: str
    parent_subgraph: int
    embedding: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.embedding = get_embedding(self.label)

    def __repr__(self) -> str:
        return f"Edge({self.agent_1} <--[{self.label}]--> {self.agent_2})"

    def __eq__(self, value):
        return (
            self.agent_1 == value.agent_1
            and self.agent_2 == value.agent_2
            and self.label == value.label
        )

    def __hash__(self) -> int:
        return hash((self.agent_1, self.agent_2, self.label))

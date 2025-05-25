import pickle

from collections import defaultdict
from typing import Dict, List, Optional, Set

from graph.vertex import Vertex
from graph.edge import Edge
from graph.union_edge import UnionEdge

dictionary = set()
with open("embeddings/dictionary.txt", "rt", encoding="utf-8") as f:
    for line in f:
        dictionary.add(line.strip())


class Graph:
    """
    Represents a graph structure with vertices and edges, supporting union edges.
    """

    def __init__(self) -> None:
        self.vertices: Dict[str, Vertex] = {}
        self.edges: Set[Edge] = set()
        self.edge_ids: Dict[Edge, int] = {}
        self.edge_by_ids: Dict[int, Edge] = {}
        self.union_edges: List[UnionEdge] = []
        self.vertex_edges: Dict[str, Set[int]] = defaultdict(set)

    def __repr__(self) -> str:
        return (
            f"Graph(\n\tvertices={list(self.vertices.values())},\n"
            f"\tedges={self.edges}\n)"
        )

    def add_vertex(self, concept: str) -> None:
        """
        Add a vertex to the graph.

        Args:
            concept: The main concept for the vertex.
        """

        if concept in self.vertices:
            return

        self.vertices[concept] = Vertex(concept)

    def add_edge(
        self,
        agent_1: str,
        agent_2: str,
        label: str,
        parent_subgraph: int = 1,
    ) -> None:
        """
        Add an edge between two vertices.

        Args:
            agent_1: Source vertex concept.
            agent_2: Target vertex concept.
            label: The relationship meaning.
            parent_subgraph: ID of the parent text.
        """

        edge = Edge(agent_1, agent_2, label, parent_subgraph)

        if edge in self.edges:
            return

        edge_id = len(self.edges)

        self.edges.add(edge)
        self.edge_ids[edge] = edge_id
        self.edge_by_ids[edge_id] = edge

        self.vertices[agent_1].adjacent_edges.add(edge_id)
        self.vertices[agent_2].adjacent_edges.add(edge_id)

        self.vertex_edges[agent_1].add(edge_id)
        self.vertex_edges[agent_2].add(edge_id)

    def add_individual_edges(
        self, group: List[Edge], edge_ids: Optional[Set] = None
    ) -> Optional[Set]:
        """
        Add a group of edges to the graph.

        Args:
            group: List of edges to add.
            edge_ids: Set of IDs of added edges.

        Returns:
            Set of IDs of added edges.
        """

        for edge in group:
            edge_id = len(self.edges)

            if (
                edge.agent_1 not in dictionary
                or edge.agent_2 not in dictionary
                or edge.label not in dictionary
            ):
                continue

            if edge.agent_1 not in self.vertices:
                self.add_vertex(edge.agent_1)
            if edge.agent_2 not in self.vertices:
                self.add_vertex(edge.agent_2)

            if edge not in self.edges:
                self.add_edge(
                    edge.agent_1, edge.agent_2, edge.label, edge.parent_subgraph
                )
                if edge_ids is not None:
                    edge_ids.add(edge_id)

            elif edge_ids is not None:
                edge_ids.add(self.edge_ids[edge])

        return edge_ids

    def add_union_edge_many2one(
        self,
        agent_group_1: List[Edge],
        agent_2: str,
        label: str,
        parent_subgraph: int = 1,
    ) -> None:
        """
        Add edges representing a relation with multiple parts.

        Args:
            agent_group_1: List of tuples representing the first part of the higher dimensional graph.
            agent_2: The second part of the higher dimensional graph.
            label: The relationship meaning.
            parent_subgraph: ID of the parent text.
        """

        edge_ids = set()
        edge_ids = self.add_individual_edges(agent_group_1, edge_ids)

        if not edge_ids or agent_2 not in dictionary or label not in dictionary:
            return

        if agent_2 not in self.vertices:
            self.add_vertex(agent_2)

        for edge in agent_group_1:
            edge_ids = self.add_individual_edges(
                [
                    Edge(agent, agent_2, label, parent_subgraph)
                    for agent in (edge.agent_1, edge.agent_2)
                ],
                edge_ids,
            )

        union_edge = UnionEdge(
            edge_ids=edge_ids,
            label=label,
            parent_subgraph=parent_subgraph,
        )
        self.union_edges.append(union_edge)

    def add_union_edge_one2many(
        self,
        agent_1: str,
        agent_group_2: List[Edge],
        label: str,
        parent_subgraph: int = 1,
    ) -> None:
        """
        Add edges representing a relation with multiple parts.

        Args:
            agent_1: The first part of the higher dimensional graph.
            agent_group_2: List of tuples representing the second part of the higher dimensional graph.
            label: The relationship meaning.
            parent_subgraph: ID of the parent text.
        """

        edge_ids = set()
        edge_ids = self.add_individual_edges(agent_group_2, edge_ids)

        if not edge_ids or agent_1 not in dictionary or label not in dictionary:
            return

        if agent_1 not in self.vertices:
            self.add_vertex(agent_1)

        for edge in agent_group_2:
            edge_ids = self.add_individual_edges(
                [
                    Edge(agent_1, agent, label, parent_subgraph)
                    for agent in (edge.agent_1, edge.agent_2)
                ],
                edge_ids,
            )

        union_edge = UnionEdge(
            edge_ids=edge_ids,
            label=label,
            parent_subgraph=parent_subgraph,
        )
        self.union_edges.append(union_edge)

    def add_union_edge_many2many(
        self,
        agent_group_1: List[Edge],
        agent_group_2: List[Edge],
        label: str,
        parent_subgraph: int = 1,
    ) -> None:
        """
        Add edges representing a relation with multiple parts.

        Args:
            agent_group_1: List of tuples representing the first part of the higher dimensional graph.
            agent_group_2: List of tuples representing the second part of the higher dimensional graph.
            label: Thr relationship meaning.
            parent_subgraph: ID of the parent text.
        """

        edge_ids = set()
        edge_ids = self.add_individual_edges(agent_group_1, edge_ids)

        if not edge_ids:
            return

        edge_ids = self.add_individual_edges(agent_group_2, edge_ids)

        if not edge_ids or label not in dictionary:
            return

        for edge_1 in agent_group_1:
            for edge_2 in agent_group_2:
                for agent_1 in (edge_1.agent_1, edge_1.agent_2):
                    edge_ids = self.add_individual_edges(
                        [
                            Edge(agent_1, agent, label, parent_subgraph)
                            for agent in (edge_2.agent_1, edge_2.agent_2)
                        ],
                        edge_ids,
                    )

        union_edge = UnionEdge(
            edge_ids=edge_ids,
            label=label,
            parent_subgraph=parent_subgraph,
        )
        self.union_edges.append(union_edge)

    def get_union_edges(self) -> List[Dict]:
        """
        Get all union edges with their component edges.

        Returns:
            List of dictionaries containing information about union edges.
        """

        result = []

        for union_edge in self.union_edges:
            edges = [self.edge_by_ids[i] for i in union_edge.edge_ids]
            result.append(
                {
                    "edges": edges,
                    "label": union_edge.label,
                    "parent_subgraph": union_edge.parent_subgraph,
                }
            )

        return result


def save_graph(graph: Graph, filename: str) -> None:
    with open(filename, "wb") as f:
        pickle.dump(graph, f)


def load_graph(filename: str) -> Graph:
    with open(filename, "rb") as f:
        return pickle.load(f)

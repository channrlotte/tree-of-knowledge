import numpy as np
import random

from typing import Dict, List, Set
from tqdm import tqdm

from graph.higher_dim_graph import Graph, save_graph, load_graph
from graph.vertex import Vertex


def cluster_vertices(vertices: List[Vertex]) -> Dict[str, str]:
    random.shuffle(vertices)

    unused_concepts: Set[str] = {v.concept for v in vertices}
    clusters: Dict[str, list[Vertex]] = {v.concept: [v] for v in vertices}
    vertices_to_clusters: Dict[str, str] = {v.concept: v.concept for v in vertices}

    def dist_to_group(v: Vertex, cluster: str) -> float:
        return min(
            np.dot(v.embedding, cluster_v.embedding)
            / np.linalg.norm(v.embedding)
            / np.linalg.norm(cluster_v.embedding)
            for cluster_v in clusters[cluster]
        )

    for v in tqdm(vertices):
        if v.concept not in unused_concepts:
            continue
        unused_concepts.remove(v.concept)

        if len(unused_concepts) == 0 or len(clusters[v.concept]) > 1:
            continue

        dist = lambda x: dist_to_group(v, x)
        nearest = max(unused_concepts, key=dist)
        if dist(nearest) < 0.6:
            continue

        clusters[nearest].append(v)
        vertices_to_clusters[v.concept] = nearest
        if len(clusters[nearest]) >= 12:
            unused_concepts.remove(nearest)

    for cluster in clusters:
        if len(clusters[cluster]) > 0:
            clusters[cluster] = sorted(
                list(clusters[cluster]),
                key=lambda x: sum(
                    np.linalg.norm(cluster_v.embedding - x.embedding)
                    for cluster_v in clusters[cluster]
                ),
            )

    vertices_to_clusters = {
        vertex: clusters[cluster][0].concept
        for vertex, cluster in vertices_to_clusters.items()
    }

    return vertices_to_clusters


def squeeze_graph(file_name: str) -> None:
    graph = load_graph(f"data/graphs/higher_dim_graphs/{file_name}.pickle")

    vertices = set(graph.vertices.values())
    vertices_to_clusters = cluster_vertices(list(vertices))

    clustered_graph = Graph()
    added_edges = set()

    for cluster_label in set(vertices_to_clusters.values()):
        clustered_graph.add_vertex(cluster_label)

    for edge in tqdm(graph.edges):
        new_edge = (
            vertices_to_clusters.get(edge.agent_1, edge.agent_1),
            vertices_to_clusters.get(edge.agent_2, edge.agent_2),
            edge.label,
        )

        if new_edge in added_edges:
            continue
        added_edges.add(new_edge)

        if new_edge[0] not in clustered_graph.vertices:
            clustered_graph.add_vertex(new_edge[0])
        if new_edge[1] not in clustered_graph.vertices:
            clustered_graph.add_vertex(new_edge[1])

        clustered_graph.add_edge(*new_edge)

    save_graph(
        clustered_graph, f"data/graphs/clustered_graphs/clustered_{file_name}.pickle"
    )


if __name__ == "__main__":
    squeeze_graph("data/graphs/higher_dim_graphs/articles.pickle")
    squeeze_graph("data/graphs/higher_dim_graphs/fiction.pickle")

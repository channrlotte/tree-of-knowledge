import glob
import json

from tqdm import tqdm

from graph.higher_dim_graph import Graph, save_graph, load_graph
from graph.edge import Edge


def build_graph(directory_name: str) -> None:
    graph = Graph()

    for file_id, file_name in tqdm(
        enumerate(sorted(glob.glob(f"data/relations/{directory_name}/*.json"))),
        total=len(glob.glob(f"data/relations/{directory_name}/*.json")),
    ):
        with open(file_name, "rt", encoding="utf-8") as f:
            relations = json.load(f)

        for relation in relations:
            if relation[0] == 1:
                graph.add_individual_edges(
                    [Edge(relation[1], relation[3], relation[2], file_id)]
                )

            elif relation[0] == 2:
                graph.add_individual_edges(
                    [Edge(relation[1][0], relation[1][2], relation[1][1], file_id)]
                )
                graph.add_union_edge_many2one(
                    [Edge(relation[1][0], relation[1][2], relation[1][1], file_id)],
                    relation[3],
                    relation[2],
                    file_id,
                )

            elif relation[0] == 3:
                graph.add_individual_edges(
                    [Edge(relation[3][0], relation[3][2], relation[3][1], file_id)]
                )
                graph.add_union_edge_one2many(
                    relation[1],
                    [Edge(relation[3][0], relation[3][2], relation[3][1], file_id)],
                    relation[2],
                    file_id,
                )

            elif relation[0] == 4:
                graph.add_individual_edges(
                    [Edge(relation[1][0], relation[1][2], relation[1][1], file_id)]
                )
                graph.add_individual_edges(
                    [Edge(relation[3][0], relation[3][2], relation[3][1], file_id)]
                )
                graph.add_union_edge_many2many(
                    [Edge(relation[1][0], relation[1][2], relation[1][1], file_id)],
                    [Edge(relation[3][0], relation[3][2], relation[3][1], file_id)],
                    relation[2],
                    file_id,
                )

    save_graph(graph, f"data/graphs/higher_dim_graphs/{directory_name}.pickle")


if __name__ == "__main__":
    build_graph("mother")

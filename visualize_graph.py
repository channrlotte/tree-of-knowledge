import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

from graph.higher_dim_graph import Graph, load_graph
from metrics import detect_hubs_with_metrics


def visualize_graph_2d(graph: Graph, top_n: int) -> go.Figure:
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(
        (edge.agent_1, edge.agent_2, {"label": edge.label}) for edge in graph.edges
    )
    pos = nx.spring_layout(nx_graph)

    hubs = detect_hubs_with_metrics(graph, top_n=top_n)
    hub_nodes = [n for n in nx_graph.nodes if n in hubs]
    other_nodes = [n for n in nx_graph.nodes if n not in hubs]

    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=hub_nodes,
        node_color="midnightblue",
        node_size=2500,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=other_nodes,
        node_color="lightsteelblue",
        node_size=1500,
        alpha=0.7,
    )

    nx.draw_networkx_edges(nx_graph, pos, edge_color="gray")

    nx.draw_networkx_labels(nx_graph, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(nx_graph, "label")
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels, font_size=7)

    plt.title("Визуализация графа в 2D пространстве")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_graph_3d(graph: Graph, top_n: int) -> go.Figure:
    hubs = detect_hubs_with_metrics(graph, top_n=top_n)

    x_nodes, y_nodes, z_nodes = [], [], []
    node_labels = []
    node_colors = []

    for concept, vertex in graph.vertices.items():
        if vertex.embedding is None:
            continue
        if vertex.embedding.shape[0] < 3:
            raise ValueError("Vertex embeddings must be at least 3D for visualization.")

        x_nodes.append(vertex.embedding[0])
        y_nodes.append(vertex.embedding[1])
        z_nodes.append(vertex.embedding[2])

        node_labels.append(concept)
        node_colors.append("midnightblue" if concept in hubs else "lightsteelblue")

    edge_x, edge_y, edge_z = [], [], []
    edge_labels, edge_label_x, edge_label_y, edge_label_z = [], [], [], []

    for edge in graph.edges:
        agent_1, agent_2 = graph.vertices[edge.agent_1], graph.vertices[edge.agent_2]

        if agent_1.embedding is None or agent_2.embedding is None:
            continue

        edge_x.extend([agent_1.embedding[0], agent_2.embedding[0], None])
        edge_y.extend([agent_1.embedding[1], agent_2.embedding[1], None])
        edge_z.extend([agent_1.embedding[2], agent_2.embedding[2], None])

        edge_labels.append(edge.label)
        edge_label_x.append((agent_1.embedding[0] + agent_2.embedding[0]) / 2)
        edge_label_y.append((agent_1.embedding[1] + agent_2.embedding[1]) / 2)
        edge_label_z.append((agent_1.embedding[2] + agent_2.embedding[2]) / 2)

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2, color="gray"),
    )

    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers+text",
        marker=dict(size=7, color=node_colors, opacity=0.9),
        text=node_labels,
        textfont=dict(size=10),
        textposition="top center",
    )

    edge_label_trace = go.Scatter3d(
        x=edge_label_x,
        y=edge_label_y,
        z=edge_label_z,
        mode="text",
        text=edge_labels,
        textfont=dict(size=7),
        textposition="middle center",
    )

    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(
        title="Визуализация графа в 3D пространстве",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        showlegend=False,
    )
    fig.update_layout(width=1000, height=1000)
    fig.show()


if __name__ == "__main__":
    graph = load_graph("data/graphs/higher_dim_graphs/mother.pickle")
    visualize_graph_2d(graph, top_n=1)
    visualize_graph_3d(graph, top_n=1)

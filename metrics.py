import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.stats import mannwhitneyu

from typing import Any, List
from tqdm import tqdm

from graph.higher_dim_graph import Graph, load_graph


def convert_to_networkx(graph: Graph) -> nx.MultiGraph:
    nx_graph = nx.MultiGraph()
    nx_graph.add_edges_from(
        (edge.agent_1, edge.agent_2, {"label": edge.label}) for edge in graph.edges
    )

    return nx_graph


def show_distribution_comparison(
    values1: List[float],
    values2: List[float],
    title: str,
    xlabel: str,
    ylabel: str = "Частота",
    log_y: bool = False,
    bins: Any = 20,
    labels: tuple = ("Научный", "Художественный"),
    colors: tuple = ("lightsteelblue", "midnightblue"),
) -> None:

    combined = values1 + values2

    const = 2
    ticks = False

    if isinstance(bins, int):
        bins = np.linspace(min(combined), max(combined), bins)
    else:
        const = 4
        ticks = True

    hist1, _ = np.histogram(values1, bins=bins)
    hist2, _ = np.histogram(values2, bins=bins)

    hist1 = hist1 / sum(hist1)
    hist2 = hist2 / sum(hist2)

    width = np.diff(bins)
    centers = bins[:-1]

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()

    ax.bar(
        centers - width / (2 * const),
        hist1,
        width=width / const,
        align="center",
        label=labels[0],
        color=colors[0],
        edgecolor="black",
        alpha=0.7,
    )
    ax.bar(
        centers + width / (2 * const),
        hist2,
        width=width / const,
        align="center",
        label=labels[1],
        color=colors[1],
        edgecolor="black",
        alpha=0.7,
    )

    if ticks:
        ax.set_xticks(centers)
        ax.set_xticklabels([str(i) for i in centers])

    if log_y:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="-", alpha=0.5)
    plt.show()


def check_criterion(
    values1: List[float], values2: List[float], distribution: str
) -> None:
    stat, p = mannwhitneyu(values1, values2, alternative="two-sided")

    if p < 0.05:
        print(f"{distribution}: Различия статистически значимы (p={p} < 0.05)")
    else:
        print(f"{distribution}: Нет статистически значимых различий (p={p} ≥ 0.05)")


def compare_degrees(nx_graph1: nx.MultiGraph, nx_graph2: nx.MultiGraph) -> None:
    degrees1 = [deg for _, deg in nx_graph1.degree()]
    degrees2 = [deg for _, deg in nx_graph2.degree()]

    show_distribution_comparison(
        degrees1, degrees2, title="Распределение степеней", xlabel="Степень", log_y=True
    )

    check_criterion(degrees1, degrees2, "Распределение степеней")


def compare_clustering(nx_graph1: nx.MultiGraph, nx_graph2: nx.MultiGraph) -> None:
    clustering1 = list(nx.clustering(nx.Graph(nx_graph1)).values())
    clustering2 = list(nx.clustering(nx.Graph(nx_graph2)).values())

    show_distribution_comparison(
        clustering1,
        clustering2,
        title="Распределение коэффициентов кластеризации",
        xlabel="Коэффициент кластеризации",
    )

    check_criterion(
        clustering1, clustering2, "Распределение коэффициентов кластеризации"
    )


def get_path_lengths(nx_graph: nx.MultiGraph) -> List[float]:
    all_pairs_shortest_paths = dict(nx.all_pairs_shortest_path_length(nx_graph))
    path_lengths: List[float] = []

    for source, targets in all_pairs_shortest_paths.items():
        for target, length in targets.items():
            if source != target:
                path_lengths.append(length)

    return path_lengths


def compare_paths(nx_graph1: nx.MultiGraph, nx_graph2: nx.MultiGraph) -> None:
    paths1 = get_path_lengths(nx_graph1)
    paths2 = get_path_lengths(nx_graph2)

    show_distribution_comparison(
        paths1,
        paths2,
        title="Распределение длин кратчайших путей",
        xlabel="Длина кратчайшего пути",
        log_y=True,
        bins=np.arange(1, 9),
    )

    check_criterion(paths1, paths2, "Распределение длин кратчайших путей")


def compare_closeness(nx_graph1: nx.MultiGraph, nx_graph2: nx.MultiGraph) -> None:
    closeness1 = list(nx.closeness_centrality(nx_graph1).values())
    closeness2 = list(nx.closeness_centrality(nx_graph2).values())

    show_distribution_comparison(
        closeness1,
        closeness2,
        title="Распределение степеней близости",
        xlabel="Степень близости",
        log_y=True,
    )

    check_criterion(closeness1, closeness2, "Распределение степеней близости")


def compare_betweenness(nx_graph1: nx.MultiGraph, nx_graph2: nx.MultiGraph) -> None:
    betweenness1 = list(nx.betweenness_centrality(nx_graph1, k=500).values())
    betweenness2 = list(nx.betweenness_centrality(nx_graph2, k=500).values())

    show_distribution_comparison(
        betweenness1,
        betweenness2,
        title="Распределение степеней посредничества",
        xlabel="Степень посредничества",
        log_y=True,
    )

    check_criterion(betweenness1, betweenness2, "Распределение степеней посредничества")


def compare_components(nx_graph1: nx.MultiGraph, nx_graph2: nx.MultiGraph) -> None:
    components1 = list(nx.connected_components(nx_graph1))
    components2 = list(nx.connected_components(nx_graph2))
    component_sizes1 = [len(component) for component in components1]
    component_sizes2 = [len(component) for component in components2]

    show_distribution_comparison(
        component_sizes1,
        component_sizes2,
        title="Распределение размеров компонент",
        xlabel="Размеры компонент",
        log_y=True,
    )

    check_criterion(
        component_sizes1, component_sizes2, "Распределение размеров компонент"
    )


def assortativity(nx_graph: nx.MultiGraph) -> float:
    return nx.degree_assortativity_coefficient(nx_graph)


def detect_hubs_with_metrics(
    graph: Graph,
    degree_weight: float = 0.4,
    betweenness_weight: float = 0.3,
    closeness_weight: float = 0.3,
    threshold: float = None,
    top_n: int = None,
) -> List[str]:
    nx_graph = convert_to_networkx(graph)

    if threshold is not None and top_n is not None:
        raise ValueError("Only one of 'threshold' or 'top_n' can be specified.")

    degree_centrality = nx.degree_centrality(nx.Graph(nx_graph))
    betweenness_centrality = nx.betweenness_centrality(nx_graph)
    closeness_centrality = nx.closeness_centrality(nx_graph)

    total_weight = betweenness_weight + degree_weight + closeness_weight
    degree_weight /= total_weight
    betweenness_weight /= total_weight
    closeness_weight /= total_weight

    composite_scores = {}
    for node in tqdm(nx_graph.nodes):
        degree_score = degree_centrality[node]
        betweenness_score = betweenness_centrality[node]
        closeness_score = closeness_centrality[node]

        composite_score = (
            degree_weight * degree_score
            + betweenness_weight * betweenness_score
            + closeness_weight * closeness_score
        )
        composite_scores[node] = composite_score

    if threshold is not None:
        hubs = [node for node, score in composite_scores.items() if score >= threshold]

    elif top_n is not None:
        sorted_nodes = sorted(
            composite_scores.items(), key=lambda x: x[1], reverse=True
        )
        hubs = [node for node, score in sorted_nodes[:top_n]]

    else:
        raise ValueError("Either 'threshold' or 'top_n' must be specified.")

    return hubs


if __name__ == "__main__":
    graph1 = load_graph("data/graphs/clustered_graphs/clustered_articles.pickle")
    graph2 = load_graph("data/graphs/clustered_graphs/clustered_fiction.pickle")
    nx_graph1 = convert_to_networkx(graph1)
    nx_graph2 = convert_to_networkx(graph2)

    compare_degrees(nx_graph1, nx_graph2)
    compare_clustering(nx_graph1, nx_graph2)
    compare_paths(nx_graph1, nx_graph2)
    compare_closeness(nx_graph1, nx_graph2)
    compare_betweenness(nx_graph1, nx_graph2)
    compare_components(nx_graph1, nx_graph2)

    print("Ассортативность в научном тексте: ", assortativity(nx_graph1))
    print("Ассортативность в художественном тексте: ", assortativity(nx_graph2))

    print(
        "Ключевые концепции в научном тексте: ",
        ", ".join(detect_hubs_with_metrics(graph1, top_n=10)),
    )
    print(
        "Ключевые концепции в художественном тексте: ",
        ", ".join(detect_hubs_with_metrics(graph2, top_n=10)),
    )

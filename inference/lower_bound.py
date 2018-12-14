from random import sample
from typing import List

import networkx as nx

from core.mallows import Mallows
from core.patterns import Pattern, BipartitePattern
from inference.infer_bipartite import BipartiteInferencer


def calculate_dag_from_pattern(pattern: Pattern):
    label_to_item = {label: sample(items, k=1)[0] for label, items in pattern.label_to_items.items()}
    graph = nx.DiGraph()
    for label, children in pattern.label_to_children.items():
        parent_item = label_to_item[label]
        for child in children:
            child_item = label_to_item[child]
            graph.add_edge(parent_item, child_item)

    if nx.is_directed_acyclic_graph(graph):
        return graph
    else:
        return calculate_dag_from_pattern(pattern)


def calculate_lower_bound_bipartite_pattern(pattern: Pattern, center: List):
    dag = calculate_dag_from_pattern(pattern)

    l_labels, r_labels = set(), set()
    label_to_children = {}
    label_to_items = {}
    for l, r in dag.edges:
        l_name, r_name = f'L-{l}', f'R-{r}'

        if l_name not in l_labels:
            l_labels.add(l_name)
            label_to_items[l_name] = {l}

        if r_name not in r_labels:
            r_labels.add(r_name)
            label_to_items[r_name] = {r}

        label_to_children.setdefault(l_name, set()).add(r_name)

    return BipartitePattern(label_to_children, label_to_items)


def calculate_lower_bound(patterns: List[Pattern], mallows: Mallows):
    bipartite_patterns = [calculate_lower_bound_bipartite_pattern(pattern, mallows.center) for pattern in patterns]
    bipartite_inferencer = BipartiteInferencer(mallows)
    res = bipartite_inferencer.solve(bipartite_patterns)
    return res

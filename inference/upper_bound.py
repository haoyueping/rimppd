from typing import List

from core.mallows import Mallows
from core.patterns import Pattern, BipartitePattern
from inference.infer_bipartite import BipartiteInferencer


def calculate_upper_bound_bipartite_pattern(pattern: Pattern):
    pattern.calculate_tc()

    l_labels, r_labels = set(), set()
    label_to_children = {}
    label_to_items = {}
    for l, r in pattern.tc.edges:
        l_name, r_name = f'L-{l}', f'R-{r}'

        if l_name not in l_labels:
            l_labels.add(l_name)
            label_to_items[l_name] = pattern.get_items_in_label(l)

        if r_name not in r_labels:
            r_labels.add(r_name)
            label_to_items[r_name] = pattern.get_items_in_label(r)

        label_to_children.setdefault(l_name, set()).add(r_name)

    return BipartitePattern(label_to_children, label_to_items)


def calculate_upper_bound(patterns: List[Pattern], mallows: Mallows):
    bipartite_patterns = [calculate_upper_bound_bipartite_pattern(pattern) for pattern in patterns]
    bipartite_inferencer = BipartiteInferencer(mallows)
    res = bipartite_inferencer.solve(bipartite_patterns)
    return res

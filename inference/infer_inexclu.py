from copy import deepcopy
from itertools import combinations
from typing import List, Tuple, Iterable

from core.mallows import Mallows
from core.patterns import Pattern
from inference.ltm.ltm_wrapper import calculate_marginal_prob_over_mallows_by_ltm


def generate_a_large_pattern_from_sub_patterns(patterns: Iterable[Pattern]):
    label_to_children = {}
    label_to_items = {}

    for idx, pattern in enumerate(patterns):

        for parent, children in pattern.label_to_children.items():
            parent_name = f'{idx}-{parent}'
            label_to_children[parent_name] = set()
            for child in children:
                label_to_children[parent_name].add(f'{idx}-{child}')

        for label, items in pattern.label_to_items.items():
            label_to_items[f'{idx}-{label}'] = items

    return Pattern(label_to_children, label_to_items)


def calculate_patterns_over_mallows_by_inexclu(patterns: List[Pattern], mallows: Mallows) -> Tuple[float, int]:
    pattern_list = deepcopy(patterns)

    prob_triangle = []
    t_final = 0
    for size in range(1, len(pattern_list) + 1):
        prob_list = []
        for patterns_tup in combinations(pattern_list, size):
            pattern_comb = generate_a_large_pattern_from_sub_patterns(patterns_tup)
            _, prob, t = calculate_marginal_prob_over_mallows_by_ltm(pattern_comb, mallows)

            prob_list.append(prob)
            t_final += t

        prob_triangle.append(prob_list)

    p_final = 0
    symbol = 1
    for prob_list in prob_triangle:
        p_final += sum(prob_list) * symbol
        symbol *= -1

    return p_final, t_final


if __name__ == '__main__':

    from experiment_code.utils import generate_random_bipartite_pattern_ab_ac
    from deprecated.infer_stars_deprecated import StarInferencer

    m = 5
    phi = 0.6

    num_patterns = 5
    label_size = 3

    mallows = Mallows(list(range(m)), phi)

    pattern_list = [generate_random_bipartite_pattern_ab_ac(m, label_size, 0.3) for _ in range(num_patterns)]

    for pattern in pattern_list:
        print(pattern)

    star_inferencer = StarInferencer(mallows)
    p_star, t_star = star_inferencer.solve(pattern_list)

    print(p_star)

    p_inexclu, t_inexclu = calculate_patterns_over_mallows_by_inexclu(pattern_list, mallows)

    print(p_star, p_inexclu)
    print(t_star, t_inexclu)

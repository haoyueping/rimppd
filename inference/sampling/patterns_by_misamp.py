from typing import List

from core.mallows import Mallows
from core.patterns import Pattern
from inference.sampling.pattern_by_misamp import decompose_pattern, estimate_union_of_prefs


def estimate_patterns_by_misamp(patterns: List[Pattern], mallows: Mallows, threshold=0.01, single_core_workload=50,
                                num_cores=None, verbose=False):
    for pattern in patterns:
        pattern.calculate_tc()

    seeds = set()
    for pattern in patterns:
        seeds.update(decompose_pattern(pattern))

    return estimate_union_of_prefs(seeds, mallows, threshold, single_core_workload, num_cores, verbose)


def test_movielens():
    from inference.sampling.utils import get_test_case_of_patterns_from_movielens_5_labels
    from deprecated.patterns_by_rejection import estimate_patterns_by_rejection

    for rid in [36, 52, 68, 84, 100, 116, 132, 148]:
        patterns, mallows = get_test_case_of_patterns_from_movielens_5_labels(rid)
        res = estimate_patterns_by_misamp(patterns, mallows)
        res_rejection = estimate_patterns_by_rejection(patterns, mallows)
        print('MIS-AMP1:', res, '\nrejection:', res_rejection)
        break


def test_4_labels(pid=2):
    from inference.sampling.utils import get_test_case_of_patterns_from_synthetic_4_labels

    patterns, mallows, p_exact = get_test_case_of_patterns_from_synthetic_4_labels(pid)
    print(f'p_exact = {p_exact}')
    res = estimate_patterns_by_misamp(mallows=mallows, patterns=patterns, verbose=True)
    print(res)
    print(f'p_exact = {p_exact}')


if __name__ == '__main__':
    # from random import seed
    # from numpy.random import seed as npseed
    #
    # seed(0)
    # npseed(0)

    test_4_labels(0)

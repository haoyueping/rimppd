from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from time import time
from typing import List

from numpy import prod
from numpy.random import choice

from core.mallows import Mallows
from core.patterns import Pattern, ItemPref
from helpers.utils import normalize_numeric_list
from inference.sampling.itempref_by_isamp import amp_sampler
from inference.sampling.pattern_by_misamp import prob_of_ranking_from_pref_by_amp
from inference.sampling.pattern_by_misamp1 import sample_seed_pref


def estimate_prob_over_all_seeds(r, patterns, mallows, pattern_probs, k=100):
    item_to_rank = {item: rank for rank, item in enumerate(r)}
    prob_sum = 0
    for _ in range(k):
        seed_pref = sample_seed_pref(choice(patterns, p=pattern_probs))
        if isinstance(seed_pref, ItemPref) and seed_pref.is_a_linear_extension_by_item_ranks(item_to_rank):
            prob_sum += prob_of_ranking_from_pref_by_amp(item_to_rank, seed_pref, mallows)

    return prob_sum / k


def worker(paras):
    patterns, mallows, pattern_probs, single_core_workload, k = paras

    prob_sum, prob_max = 0, 0
    for _ in range(single_core_workload):
        seed_pref = sample_seed_pref(choice(patterns, p=pattern_probs))
        if isinstance(seed_pref, ItemPref):
            r, dist, _ = amp_sampler(seed_pref, mallows)
            prob_origin = mallows.calculate_prob_by_distance(dist)
            prob_seed_satisfying_r = estimate_prob_over_all_seeds(r, patterns, mallows, pattern_probs, k)

            if prob_seed_satisfying_r > 0:
                prob_i = prob_origin / prob_seed_satisfying_r
                prob_sum += prob_i
                prob_max = max(prob_max, prob_i)

    return prob_sum, prob_max


def calculate_pattern_decomposition_size(pattern: Pattern):
    label_sizes = [len(items) for items in pattern.label_to_items.values()]
    return prod(label_sizes)


def estimate_patterns_by_misamp1(patterns: List[Pattern], mallows: Mallows, k=100, threshold=0.01, single_core_workload=50,
                                 num_cores=None, verbose=False):
    num_cores = num_cores or cpu_count()
    round_size = single_core_workload * num_cores

    patterns = deepcopy(patterns)
    pattern_probs = [calculate_pattern_decomposition_size(pattern) for pattern in patterns]
    pattern_probs = normalize_numeric_list(pattern_probs)

    if verbose:
        print('\nPatterns:')
        for pattern in patterns:
            print('  ', pattern)
        print(f'pattern_probs: {pattern_probs}')
        print(mallows, '\n')

    para_tuple = (patterns, mallows, pattern_probs, single_core_workload, k)
    prob_max, prob_sum, round_i, start_time = 0, 0, 0, time()
    while True:
        round_i += 1
        num_samples = round_i * round_size

        with Pool(processes=num_cores) as pool:
            res_list = pool.map(worker, [para_tuple for _ in range(num_cores)])

        for (prob_sum_i, prob_max_i) in res_list:
            prob_sum += prob_sum_i
            prob_max = max(prob_max, prob_max_i)

        prob_now = prob_sum / num_samples

        if verbose:
            print(f"prob={prob_now}, #samples={num_samples}, convergence={prob_max / prob_sum}")

        if prob_max < threshold * prob_sum:
            runtime = int((time() - start_time) * 1000)
            return prob_now, num_samples, runtime


def test_movielens():
    from inference.sampling.utils import get_test_case_of_patterns_from_movielens_5_labels
    from deprecated.patterns_by_rejection import estimate_patterns_by_rejection

    for rid in [36, 52, 68, 84, 100, 116, 132, 148]:
        patterns, mallows = get_test_case_of_patterns_from_movielens_5_labels(rid)
        res = estimate_patterns_by_misamp1(patterns, mallows)
        res_rejection = estimate_patterns_by_rejection(patterns, mallows)
        print('MIS-AMP1:', res, '\nrejection:', res_rejection)
        break


def test_4_labels(pid=2):
    from inference.sampling.utils import get_test_case_of_patterns_from_synthetic_4_labels

    patterns, mallows, p_exact = get_test_case_of_patterns_from_synthetic_4_labels(pid)
    print(f'p_exact = {p_exact}')

    verbose = True
    res = estimate_patterns_by_misamp1(mallows=mallows, patterns=patterns, k=1000, verbose=verbose)
    print(res)
    res = estimate_patterns_by_misamp1(mallows=mallows, patterns=patterns, k=100, verbose=verbose)
    print(res)
    res = estimate_patterns_by_misamp1(mallows=mallows, patterns=patterns, k=10, verbose=verbose)
    print(res)
    res = estimate_patterns_by_misamp1(mallows=mallows, patterns=patterns, k=3, verbose=verbose)
    print(res)
    print(f'p_exact = {p_exact}')


if __name__ == '__main__':
    from random import seed
    from numpy.random import seed as npseed

    seed(0)
    npseed(0)
    test_4_labels(10)

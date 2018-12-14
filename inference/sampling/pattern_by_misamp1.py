from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from random import choice
from time import time
from typing import List

import networkx as nx

from core.mallows import Mallows
from core.patterns import Pattern, ItemPref
from inference.sampling.itempref_by_isamp import amp_sampler
from inference.sampling.pattern_by_misamp import prob_of_ranking_from_pref_by_amp
from inference.sampling.utils import get_test_case_of_patterns_from_synthetic_4_labels


def sample_seed_pref(pattern: Pattern):
    label_to_matching_item = {label: choice(list(items)) for label, items in pattern.label_to_items.items()}

    graph = nx.DiGraph()
    pref = {}
    for label, children in pattern.label_to_children.items():
        parent_item = label_to_matching_item[label]
        child_items = {label_to_matching_item[child] for child in children}
        pref[parent_item] = child_items
        for child_item in child_items:
            graph.add_edge(parent_item, child_item)

    if nx.is_directed_acyclic_graph(graph):
        return ItemPref(pref)
    else:
        return None


def estimate_prob_over_all_seeds(r: List, pattern: Pattern, mallows: Mallows, k=100):
    item_to_rank = {item: rank for rank, item in enumerate(r)}
    prob_sum = 0
    for _ in range(k):
        pref = sample_seed_pref(pattern)
        if isinstance(pref, ItemPref) and pref.is_a_linear_extension_by_item_ranks(item_to_rank):
            prob_sum += prob_of_ranking_from_pref_by_amp(item_to_rank, pref, mallows)

    return prob_sum / k


def worker(paras):
    pattern, mallows, single_core_workload, k = paras

    prob_sum, prob_max = 0, 0
    for _ in range(single_core_workload):
        seed_pref = sample_seed_pref(pattern)
        if isinstance(seed_pref, ItemPref):
            r, dist, _ = amp_sampler(seed_pref, mallows)
            prob_origin = mallows.calculate_prob_by_distance(dist)
            prob_r_over_all_seeds = estimate_prob_over_all_seeds(r, pattern, mallows, k)

            if prob_r_over_all_seeds > 0:
                prob_i = prob_origin / prob_r_over_all_seeds
                prob_sum += prob_i
                prob_max = max(prob_max, prob_i)

    return prob_sum, prob_max


def estimate_pattern_by_misamp1(pattern: Pattern, mallows: Mallows, k=100, threshold=0.01, single_core_workload=50,
                                num_cores=None, verbose=False):
    num_cores = num_cores or cpu_count()
    round_size = single_core_workload * num_cores

    pattern = deepcopy(pattern)
    pattern.calculate_tc()

    if verbose:
        print(f'\nPattern: {pattern}\n{mallows}\n')

    para_tuple = (pattern, mallows, single_core_workload, k)
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


def test_a_single_case():
    verbose = True
    threshold = 0
    from inference.ltm.ltm_wrapper import calculate_marginal_prob_over_mallows_by_ltm

    patterns, mallows, p_exact = get_test_case_of_patterns_from_synthetic_4_labels(2)

    mallows = Mallows(list(range(10)), 0.03)
    pattern = Pattern(label_to_children={'a': {'b'}, 'b': {'c'}}, label_to_items={'a': {4, 8}, 'b': {1, 3}, 'c': {4}})
    # pattern = patterns[0]
    # mallows = Mallows(mallows.center, 0.006)

    res_exact = calculate_marginal_prob_over_mallows_by_ltm(mallows=mallows, pattern=pattern)
    print(res_exact)

    res_samp = estimate_pattern_by_misamp1(mallows=mallows, pattern=pattern, threshold=threshold, verbose=verbose)
    print(res_samp)


if __name__ == '__main__':
    # from experiment_code.utils import generate_random_pattern
    #
    # m = 20
    # mallows = Mallows(list(range(m)), 0.1)
    # pattern = generate_random_pattern(m, num_labels=5, label_size=3)
    # print(pattern)
    # res = issamp_over_pattern(pattern, mallows)
    # print(res)

    test_a_single_case()

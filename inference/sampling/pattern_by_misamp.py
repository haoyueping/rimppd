from bisect import bisect
from itertools import product
from multiprocessing.pool import Pool
from os import cpu_count
from time import time
from typing import Set

import networkx as nx

from core.mallows import Mallows
from core.patterns import Pattern, ItemPref
from helpers.utils import normalize_numeric_list
from inference.sampling.itempref_by_isamp import amp_sampler


def prob_of_ranking_from_pref_by_amp(ranking_by_item_to_rank, pref: ItemPref, mallows):
    r = []
    ranks = []
    inserted_items = set()
    prob = 1
    for item in mallows.center:
        item_to_rank = {e: rank for rank, e in enumerate(r)}

        # TODO optimization
        inserted_ancestors = inserted_items.intersection(pref.get_all_ancestor_labels(item))
        inserted_descendants = inserted_items.intersection(pref.get_all_descendant_labels(item))

        l_indices = [item_to_rank[ancestor] for ancestor in inserted_ancestors]
        h_indices = [item_to_rank[descendant] for descendant in inserted_descendants]
        limit_low_index = max(l_indices, default=-1)
        limit_high_index = min(h_indices, default=len(r))

        insertion_range = list(range(limit_low_index + 1, limit_high_index + 1))

        item_rank_in_output = ranking_by_item_to_rank[item]
        pos_j = bisect(ranks, item_rank_in_output)
        if pos_j in insertion_range:
            probs = [mallows.phi ** (len(r) - j) for j in insertion_range]
            probs = normalize_numeric_list(probs)

            sampled_index = insertion_range.index(pos_j)
            prob *= probs[sampled_index]

            r.insert(pos_j, item)
            ranks.insert(pos_j, item_rank_in_output)
            inserted_items.add(item)
        else:
            return 0

    return prob


def worker(paras):
    prefs, mallows, single_core_workload = paras

    prob_sum, prob_max = 0, 0
    for _ in range(single_core_workload // len(prefs)):
        for pref_id, pref in enumerate(prefs):
            r, dist, prob_q_i = amp_sampler(pref, mallows)
            prob_origin = mallows.calculate_prob_by_distance(dist)

            # calculate probs of r over all prefs
            item_to_rank = {item: rank for rank, item in enumerate(r)}
            prob_over_all_prefs = 0
            for pref_id_other, pref_other in enumerate(prefs):
                if pref_id_other == pref_id:
                    prob_over_all_prefs += prob_q_i
                elif pref_other.is_a_linear_extension_by_item_ranks(item_to_rank):
                    prob_over_all_prefs += prob_of_ranking_from_pref_by_amp(item_to_rank, pref_other, mallows)

            if prob_over_all_prefs > 0:
                prob_i = prob_origin / prob_over_all_prefs * len(prefs)
                prob_sum += prob_i
                prob_max = max(prob_max, prob_i)

    return prob_sum, prob_max


def estimate_union_of_prefs(prefs: Set[ItemPref], mallows: Mallows, threshold=0.01, single_core_workload=50, num_cores=None,
                            verbose=False):
    num_cores = num_cores or cpu_count()
    single_core_workload = single_core_workload // len(prefs) * len(prefs)
    single_core_workload = single_core_workload or len(prefs)
    round_size = single_core_workload * num_cores

    para_tuple = (prefs, mallows, single_core_workload)
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


def decompose_pattern(pattern: Pattern) -> Set[ItemPref]:
    labels = list(pattern.label_to_items)
    items_list = [list(pattern.label_to_items[label]) for label in labels]

    prefs = set()
    for matching_items in product(*items_list):
        label_to_item = {label: item for label, item in zip(labels, matching_items)}

        graph = nx.DiGraph()
        pref = {}
        for (u, v) in pattern.graph.edges:
            parent_item = label_to_item[u]
            child_item = label_to_item[v]
            graph.add_edge(parent_item, child_item)
            pref.setdefault(parent_item, set()).add(child_item)

        if nx.is_directed_acyclic_graph(graph):
            prefs.add(ItemPref(pref))

    return prefs


def estimate_pattern_by_misamp(pattern: Pattern, mallows: Mallows, threshold=0.01, single_core_workload=50, num_cores=None,
                               verbose=False):
    pattern.calculate_tc()
    seeds = decompose_pattern(pattern)

    if verbose:
        print(f'\npattern: {pattern}\n{mallows}\n#seeds={len(seeds)}\n')

    return estimate_union_of_prefs(seeds, mallows, threshold, single_core_workload, num_cores, verbose)


def test_a_single_case():
    verbose = True
    threshold = 0.01
    from inference.ltm.ltm_wrapper import calculate_marginal_prob_over_mallows_by_ltm
    from inference.sampling.utils import get_test_case_of_patterns_from_synthetic_4_labels

    patterns, mallows, p_exact = get_test_case_of_patterns_from_synthetic_4_labels(2)

    # mallows = Mallows(list(range(10)), 0.3)
    # pattern = Pattern(label_to_children={'a': {'b'}}, label_to_items={'a': {5, 9}, 'b': {1, 3}})
    pattern = patterns[0]
    mallows = Mallows(mallows.center, 0.006)

    res_exact = calculate_marginal_prob_over_mallows_by_ltm(mallows=mallows, pattern=pattern)
    print(res_exact)

    res_samp = estimate_pattern_by_misamp(mallows=mallows, pattern=pattern, threshold=threshold, verbose=verbose)
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

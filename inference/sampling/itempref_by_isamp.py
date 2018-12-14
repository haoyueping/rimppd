from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from random import choices
from time import time

from core.mallows import Mallows
from core.patterns import ItemPref


def amp_sampler(pref: ItemPref, mallows: Mallows):
    r = []
    inserted_items = set()
    distance = 0
    prob = 1
    for item in mallows.center:
        item_to_rank_in_r = {e: rank for rank, e in enumerate(r)}

        inserted_ancestors = inserted_items.intersection(pref.get_all_ancestor_labels(item))
        inserted_descendants = inserted_items.intersection(pref.get_all_descendant_labels(item))

        l_indices = [item_to_rank_in_r[ancestor] for ancestor in inserted_ancestors]
        h_indices = [item_to_rank_in_r[descendant] for descendant in inserted_descendants]

        low = max(l_indices, default=-1)
        high = min(h_indices, default=len(r))

        # low + 1 because item need to be inserted before limit_low
        # high + 1 because range(x, y) outputs interval [x, y-1]
        insertion_range = list(range(low + 1, high + 1))

        i = mallows.get_rank_of_item(item)
        weights = [mallows.get_prob_i_j(i, j) for j in insertion_range]

        pos_idx = choices(list(range(len(weights))), weights=weights)[0]
        pos = insertion_range[pos_idx]
        distance += len(r) - pos
        prob *= weights[pos_idx] / sum(weights)

        r.insert(pos, item)
        inserted_items.add(item)

    return r, distance, prob


def worker(paras):
    itempref, mallows, single_core_workload = paras

    prob_sum, prob_max = 0, 0
    for _ in range(single_core_workload):
        r, dist, prob_proposal = amp_sampler(itempref, mallows)
        prob_origin = mallows.calculate_prob_by_distance(dist)
        prob_i = prob_origin / prob_proposal
        prob_sum += prob_i
        prob_max = max(prob_max, prob_i)

    return prob_sum, prob_max


def estimate_itempref_by_isamp(itempref: ItemPref, mallows: Mallows, threshold=0.01, single_core_workload=50,
                               num_cores=None, verbose=False):
    num_cores = num_cores or cpu_count()
    round_size = single_core_workload * num_cores

    itempref = deepcopy(itempref)
    itempref.calculate_tc()

    if verbose:
        print(f'\nItemPref: {itempref}\n{mallows}\n')

    para_tuple = (itempref, mallows, single_core_workload)
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


def test_a_single_case(pid=66):
    """Hard cases are 85, 66"""

    from inference.sampling.utils import get_test_case_of_itempref
    pref, mallows, p_exact = get_test_case_of_itempref(pid)

    # pref = ItemPref({4: {1}, 5: {1}})
    # mallows = Mallows(list(range(max(pref.label_to_items) + 1)), 0.05)
    #
    # from inference.ltm.ltm_wrapper import calculate_marginal_prob_over_mallows_by_ltm
    # _, p_exact, _ = calculate_marginal_prob_over_mallows_by_ltm(pref, mallows)
    print(f'p_exact={p_exact}')
    p_amp, num_samples, runtime = estimate_itempref_by_isamp(pref, mallows, threshold=0.01, single_core_workload=1000,
                                                             verbose=True)
    rela_err = round(100 * abs(p_amp - p_exact) / p_exact, 3)

    print(f'pid={pid}, p_exact={p_exact}, p_amp={p_amp} (rela_err={rela_err}%), #sample={num_samples}, time={runtime}(ms)')


if __name__ == '__main__':
    for i in range(6, 30, 3):
        print(i)
        test_a_single_case(i)

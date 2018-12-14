from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from random import choices, random
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
    itempref, mallows, single_core_workload, sample_prob, sample_factor = paras

    prob_sum, factor_sum, factor_max = 0, 0, 0
    for _ in range(single_core_workload):
        r, dist, prob_proposal = amp_sampler(itempref, mallows)
        prob_origin = mallows.calculate_prob_by_distance(dist)
        factor_i = prob_origin / prob_proposal

        if random() < factor_i / sample_factor:
            sample_prob = prob_origin
            sample_factor = factor_i

        prob_sum += sample_prob
        factor_sum += sample_factor
        factor_max = max(factor_max, sample_factor)

    return prob_sum, factor_sum, factor_max, sample_prob, sample_factor


def estimate_itempref_by_mmp(itempref: ItemPref, mallows: Mallows, threshold=0.01, single_core_workload=50,
                             num_cores=None, verbose=False):
    num_cores = num_cores or cpu_count()
    round_size = single_core_workload * num_cores

    itempref = deepcopy(itempref)
    itempref.calculate_tc()

    if verbose:
        print(f'\nItemPref: {itempref}\n{mallows}\n')

    # get the first sample
    r, dist, prob_proposal = amp_sampler(itempref, mallows)
    sample_prob = mallows.calculate_prob_by_distance(dist)
    sample_factor = sample_prob / prob_proposal
    sample_probs = [sample_prob for _ in range(num_cores)]
    sample_factors = [sample_factor for _ in range(num_cores)]

    para_tuple = (itempref, mallows, single_core_workload)
    prob_sum, factor_max, factor_sum, round_i, start_time = sample_prob, sample_factor, sample_factor, 0, time()
    while True:
        round_i += 1
        num_samples = round_i * round_size + 1

        with Pool(processes=num_cores) as pool:
            res_list = pool.map(worker, [(*para_tuple, sample_probs[idx], sample_factors[idx]) for idx in range(num_cores)])

        for idx, res_i in enumerate(res_list):
            prob_sum_i, factor_sum_i, factor_max_i, sample_prob, sample_factor = res_i
            prob_sum += prob_sum_i
            factor_sum += factor_sum_i
            factor_max = max(factor_max, factor_max_i)
            sample_probs[idx] = sample_prob
            sample_factors[idx] = sample_factor

        prob_now = prob_sum / num_samples

        if verbose:
            print(f"prob={prob_now}, #samples={num_samples}, convergence={factor_max / factor_sum}")

        if factor_max < threshold * factor_sum:
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
    p_amp, num_samples, runtime = estimate_itempref_by_mmp(pref, mallows, threshold=0.0001, single_core_workload=1000,
                                                           verbose=True)
    rela_err = round(100 * abs(p_amp - p_exact) / p_exact, 3)

    print(f'pid={pid}, p_exact={p_exact}, p_amp={p_amp} (rela_err={rela_err}%), #sample={num_samples}, time={runtime}(ms)')


if __name__ == '__main__':
    for i in range(6, 30, 3):
        print(i)
        test_a_single_case(i)

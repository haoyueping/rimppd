from collections import defaultdict
from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from random import choices
from time import time
from typing import List, Dict, Set
from warnings import warn

from core.mallows import Mallows
from core.patterns import Pattern
from deprecated.pattern_by_ramp import calculate_insertion_order_2
from inference.infer_inexclu import generate_a_large_pattern_from_sub_patterns


def calculate_insertion_order_for_ramp_over_patterns_backup(patterns: List[Pattern], center):
    """Insert shared labels first. Within each label, insert shared items first.

    E.g.,   patterns = [pattern_1, pattern_2] where patterns are essentially (1,3)>(3,5) and (1,3)>(2,5).
            center = <1,2,3,4,5,6>,
    then    output = <(3,1),(5,2),(4,6)>
    """
    label_to_count, item_to_count = defaultdict(int), defaultdict(int)

    for pattern in patterns:
        for items_in_label in pattern.label_to_items.values():

            for item in items_in_label:
                item_to_count[item] += 1

            label = '_'.join([str(item) for item in sorted(list(items_in_label))])
            label_to_count[label] += 1

    label_order = sorted(label_to_count, key=lambda x: label_to_count[x], reverse=True)
    item_order = sorted(item_to_count, key=lambda x: item_to_count[x], reverse=True)

    insertion_order = []
    for label in label_order:
        items_in_label = [int(item) for item in label.split('_')]
        items_in_label = sorted(items_in_label, key=lambda x: item_order.index(x))
        insertion_order += [item for item in items_in_label if item not in set(insertion_order)]

    insertion_order += [item for item in center if item not in set(insertion_order)]

    missing_items = set(center) - set(insertion_order)
    assert not missing_items, f'Items {missing_items} are not in the insertion order {insertion_order}.'

    return insertion_order


def calculate_insertion_order_for_isramp_over_patterns(patterns: List[Pattern], center):
    pattern_all = generate_a_large_pattern_from_sub_patterns(patterns)
    return calculate_insertion_order_2(pattern_all, center)


def ramp_sampler_over_patterns(mallows: Mallows, insertion_order: List,
                               step_to_pids_having_finishing_labels: Dict[int, Set], pid_to_step_to_sub_tc):
    pids = set(pid_to_step_to_sub_tc)
    step_to_pids_having_finishing_labels = deepcopy(step_to_pids_having_finishing_labels)

    r_sampling = []
    distance = 0
    norm = 1
    for step, item_to_insert in enumerate(insertion_order):

        insertion_range = list(range(0, len(r_sampling) + 1))
        insertion_position_to_violated_pids = defaultdict(set)

        if step in step_to_pids_having_finishing_labels:
            for pid in step_to_pids_having_finishing_labels[step]:

                sub_tc = pid_to_step_to_sub_tc[pid][step]

                for j in insertion_range:
                    r = r_sampling.copy()
                    r.insert(j, item_to_insert)

                    if not sub_tc.is_satisfying_ranking(r):
                        insertion_position_to_violated_pids[j].add(pid)

            # do not insert into position j if no sub-query can be satisfied.
            for j, violated_pids in insertion_position_to_violated_pids.items():
                if len(violated_pids) == len(pids):
                    insertion_range.remove(j)

            if not insertion_range:
                warn('Current sample fails to satisfy the query.')
                return None, None, 0

        # calculate distances and weights of all inserting positions
        item_rank_in_center = mallows.center.index(item_to_insert)
        is_later_item = [mallows.center.index(item) > item_rank_in_center for item in r_sampling]
        distances = [sum(is_later_item[:j]) + len(r_sampling) - j - sum(is_later_item[j:]) for j in insertion_range]
        weights = [mallows.phi ** dist for dist in distances]

        sampled_index_of_insertion_range = choices(range(len(insertion_range)), weights=weights)[0]

        distance += distances[sampled_index_of_insertion_range]
        norm *= sum(weights)

        insertion_position = insertion_range[sampled_index_of_insertion_range]
        r_sampling.insert(insertion_position, item_to_insert)

        if insertion_position in insertion_position_to_violated_pids:
            violated_pids = insertion_position_to_violated_pids[insertion_position]

            for pid in violated_pids:
                pids.remove(pid)

            for step_i, pids_having_finishing_labels_i in step_to_pids_having_finishing_labels.copy().items():

                if step_i > step:

                    for pid in set(pids_having_finishing_labels_i).intersection(violated_pids):
                        pids_having_finishing_labels_i.remove(pid)

                    if not pids_having_finishing_labels_i:
                        del step_to_pids_having_finishing_labels[step_i]

                else:
                    del step_to_pids_having_finishing_labels[step_i]

    return r_sampling, distance, norm


def isramp_single_thread_wrapper(para_tuple):
    norm_sum = 0
    for _ in range(para_tuple[-1]):
        _, _, norm = ramp_sampler_over_patterns(*para_tuple[:-1])
        norm_sum += norm
    return norm_sum


def isramp_over_patterns(patterns: List[Pattern], mallows: Mallows, insertion_order=None, sample_size_limit=int(1e6)):
    """IS-rAMP estimates the marginal probability of input patterns over a Mallows model.

    prob = (1/n) sum {1(x) * p(x)/q(x)} (x ~ q)
         = (1/n) sum {1(x) * (phi^d1 / Z1) / (phi^d2 / Z2)}
         = (1/n) sum {1(x) * phi^(d1-d2) * Z2 / Z1}

    :param patterns:
    :param mallows:
    :param insertion_order:
    :param sample_size_limit:
    :return:
    """
    verbose = True

    norm_origin = mallows.calculate_normalization_constant()
    insertion_order = insertion_order or calculate_insertion_order_for_isramp_over_patterns(patterns, mallows.center)

    patterns = deepcopy(patterns)

    if verbose:
        print('\nPatterns:')
        for pattern in patterns:
            print(pattern.step_to_finishing_labels)
        print(f'center ranking  = {mallows.center}')
        print(f'insertion order = {insertion_order}')

    step_to_pids_having_finishing_labels = {}
    for pid, pattern in enumerate(patterns):
        pattern.calculate_finishing_steps(insertion_order)
        for step in pattern.step_to_finishing_labels:
            step_to_pids_having_finishing_labels.setdefault(step, set()).add(pid)

    pid_to_step_to_sub_tc = {}
    for pid, pattern in enumerate(patterns):
        pid_to_step_to_sub_tc[pid] = {}
        finished_labels = set()
        for step in sorted(pattern.step_to_finishing_labels):
            finished_labels.update(pattern.step_to_finishing_labels[step])
            sub_tc = pattern.calculate_sub_tc(finished_labels)
            pid_to_step_to_sub_tc[pid][step] = sub_tc

    num_cpu = cpu_count()
    batch_size = 100  # TODO based on sampling speed
    round_size = batch_size * num_cpu

    para_tuple = (mallows, insertion_order, step_to_pids_having_finishing_labels, pid_to_step_to_sub_tc, batch_size)
    norm_ramp, prob_old, count = 0, 0, 0

    start_time = time()
    while count < sample_size_limit:

        with Pool() as pool:
            norm_sum_list = pool.map(isramp_single_thread_wrapper, [para_tuple for _ in range(num_cpu)])

        norm_ramp += sum(norm_sum_list)

        count += round_size

        prob_now = norm_ramp / norm_origin / count

        if verbose:
            print(f'prob_now = {prob_now}, #samples = {count}')

        if abs(prob_old - prob_now) < 0.001 * prob_now:
            runtime = int((time() - start_time) * 1000)
            return prob_now, count, runtime
        else:
            prob_old = prob_now

    prob = norm_ramp / norm_origin / sample_size_limit
    runtime = int((time() - start_time) * 1000)
    return prob, sample_size_limit, runtime


def test_2_label():
    import pandas as pd
    from core.patterns import PATTERN_SEP

    df_in = pd.read_csv('../../data/input_movielens_ramp-vs-amp_2labels.csv')
    df_exact = pd.read_csv('../../data/output_movielens_ramp-vs-amp_2labels_exact.csv')

    for rid in df_exact['rid']:
        p_exact = df_exact.loc[rid, 'p_exact']

        row = df_in.loc[rid]

        center_ranking = eval(row['ranking'])
        model = Mallows(center=center_ranking, phi=row['phi'])

        pattern_list = [Pattern.from_string(pattern_str) for pattern_str in row['patterns'].split(PATTERN_SEP)]

        res = isramp_over_patterns(pattern_list, model)

        print(res, f'p_exact = {p_exact}')
        break


def test_linear():
    import pandas as pd
    from core.patterns import PATTERN_SEP

    df_in = pd.read_csv('../../data/input_movielens_ramp-vs-amp.csv').tail()

    for rid, row in df_in.iterrows():
        center_ranking = eval(row['ranking'])
        model = Mallows(center=center_ranking, phi=row['phi'])

        print(f'center ranking  = {center_ranking}')

        pattern_list = [Pattern.from_string(pattern_str) for pattern_str in row['patterns'].split(PATTERN_SEP)]

        res = isramp_over_patterns(pattern_list, model)

        print(res, f'p_exact = unknown')
        break


if __name__ == '__main__':
    from random import seed
    from numpy.random import seed as npseed

    seed(0)
    npseed(0)

    test_2_label()

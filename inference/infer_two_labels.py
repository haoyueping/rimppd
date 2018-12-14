from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from time import time
from typing import List, Tuple, Dict, Set

from scipy.special import perm

from core.mallows import Mallows
from core.patterns import TwoLabelPattern


class TwoLabelInferencer(object):
    """
    It performs inference for a list of 2-label patterns over a Mallows model, i.e., calculating the probability
    that a random sampled permutation satisfies at least one pattern.

    Note that "L" is better than "H" in a permutation.
    """

    def __init__(self, mallows: Mallows):

        self.mallows = mallows

        self.patterns: List[TwoLabelPattern] = None
        self.item_to_pid_and_contributing_lh: Dict[object, Set[Tuple[int, str]]] = None
        self.step_to_pids_to_stop_tracking_lhs = None
        self.pid_to_sharing_item = {}
        self.sharing_item_to_pids = {}

        self.is_always_true = False

    def estimate_num_states_generated_during_evaluation(self, patterns: List[TwoLabelPattern]):
        self.patterns = deepcopy(patterns)

        pid_to_label_spans: Dict[int, List[List[int]]] = {}

        m = len(self.mallows.center)

        for pid, pattern in enumerate(self.patterns):

            span = [[m, -1], [m, -1]]

            for label, items in pattern.label_to_items.items():
                is_a_preferred_label = pattern.is_better_label(label)

                for item in items:
                    step = self.mallows.center.index(item)

                    if is_a_preferred_label:
                        new_l = min(span[0][0], step)
                        new_h = max(span[0][1], step)
                        span[0] = [new_l, new_h]
                    else:
                        new_l = min(span[1][0], step)
                        new_h = max(span[1][1], step)
                        span[1] = [new_l, new_h]

            new_span = [[span[0][0], span[0][1]], [span[1][0], span[1][1]]]

            pid_to_label_spans[pid] = new_span

        num_states = 0
        for step in range(m):

            num_labels = 0
            num_both_labels = 0
            for span in pid_to_label_spans.values():
                positions = 0
                if span[0][0] <= step <= span[0][1]:
                    positions += 1
                if span[1][0] <= step <= span[1][1]:
                    positions += 1

                num_labels += positions

                if positions == 2:
                    num_both_labels += 1

            num_states += perm(step + 1, len(patterns) * 2) / (2 ** len(patterns))

        return num_states

    def pre_process(self):
        for pid, pattern in enumerate(self.patterns):
            better_items = pattern.get_items_in_label(pattern.better_label)
            worse_items = pattern.get_items_in_label(pattern.worse_label)
            sharing_items = better_items.intersection(worse_items)
            if len(sharing_items) == 1:
                sharing_item = sharing_items.pop()
                self.pid_to_sharing_item[pid] = sharing_item
                self.sharing_item_to_pids.setdefault(sharing_item, set()).add(pid)
            elif len(sharing_items) == 2:
                self.is_always_true = True
                break

    def solve(self, patterns: List[TwoLabelPattern], threads=None) -> Tuple[float, int]:

        start_time = time()  # timestamp in seconds since the Epoch

        self.patterns = deepcopy(patterns)
        self.pre_process()  # calculate sharing items and self.is_always_true

        if self.is_always_true:
            return 1, 0
        else:
            self.item_to_pid_and_contributing_lh = self.calculate_item_to_pid_and_contributing_lh()
            self.step_to_pids_to_stop_tracking_lhs = self.calculate_step_to_pids_to_stop_tracking_lhs()
            max_step = max(self.step_to_pids_to_stop_tracking_lhs)

            state_to_prob: Dict[State, float] = defaultdict(float)

            threads = threads or cpu_count()

            for step, item in enumerate(self.mallows.center[:max_step + 1]):

                print(f'step {step} / {max_step}, #states = {len(state_to_prob)}')

                # initialize state_to_prob by inserting the first item.
                if step == 0:
                    init_state = State()
                    init_state = init_state.insert(step, item, 0, self)
                    state_to_prob[init_state] = 1

                # after inserting the 1st item, insert the rest items by Dynamic Programming.
                else:
                    tasks = list(state_to_prob.items())
                    pij_i = self.mallows.pij_matrix[step]
                    batched_tasks = [(tasks[i::threads], step, item, pij_i) for i in range(threads)]

                    with Pool(processes=threads) as pool:
                        res_list = pool.map(self.perform_a_batch_of_tasks, batched_tasks)

                    state_to_prob.clear()
                    for res in res_list:
                        for state, probs in res.items():
                            state_to_prob[state] += sum(probs)

            runtime_ms = int(1000 * (time() - start_time))
            return 1 - sum(state_to_prob.values()), runtime_ms

    def perform_a_batch_of_tasks(self, task_batch):

        state_and_prob_batch: Dict[State, float] = task_batch[0]
        step, item, pij_i = task_batch[1:]

        state_to_probs = defaultdict(list)

        for state, prob in state_and_prob_batch:
            for position in state.calculate_insertion_positions_iterator(step, item, self):
                state_new = state.insert(step, item, position, self)
                state_to_probs[state_new].append(prob * pij_i[position])

        return state_to_probs

    def calculate_item_to_pid_and_contributing_lh(self):
        item_to_pid_and_contributing_lh: Dict[object, Set[Tuple[int, str]]] = {}

        for pid, pattern in enumerate(self.patterns):
            sharing_item = self.pid_to_sharing_item.get(pid, None)

            for label, items in pattern.label_to_items.items():
                is_better_label = pattern.is_better_label(label)

                for item in items:
                    if item != sharing_item:

                        if is_better_label:
                            # it is positioned after the other label in negation, so it is a H bound
                            item_to_pid_and_contributing_lh.setdefault(item, set()).add((pid, 'H'))
                        else:
                            item_to_pid_and_contributing_lh.setdefault(item, set()).add((pid, 'L'))

        return item_to_pid_and_contributing_lh

    def calculate_step_to_pids_to_stop_tracking_lhs(self):
        pid_to_finishing_steps: Dict[int, List[int]] = {}

        for pid, pattern in enumerate(self.patterns):
            pid_to_finishing_steps[pid] = [-1, -1]

            for label, items in pattern.label_to_items.items():
                is_better_label = pattern.is_better_label(label)

                for item in items:
                    step = self.mallows.center.index(item)

                    if is_better_label:
                        pid_to_finishing_steps[pid][0] = max(pid_to_finishing_steps[pid][0], step)
                    else:
                        pid_to_finishing_steps[pid][1] = max(pid_to_finishing_steps[pid][1], step)

        step_to_pids_to_stop_tracking_lhs = {}

        for pid, [step_worse, step_better] in pid_to_finishing_steps.items():
            step_to_pids_to_stop_tracking_lhs.setdefault(step_worse, {}).setdefault(pid, set()).add('L')
            step_to_pids_to_stop_tracking_lhs.setdefault(step_better, {}).setdefault(pid, set()).add('H')

        return step_to_pids_to_stop_tracking_lhs


class State(object):

    def __init__(self):

        self.positions = []
        self.pid_to_lh_to_pos_rank = {}

    def __eq__(self, other):
        return isinstance(other, State) and \
               (self.positions == other.positions) and \
               (self.pid_to_lh_to_pos_rank == other.pid_to_lh_to_pos_rank)

    def __hash__(self):
        str_1 = str(self.positions)
        str_2 = str([sorted(self.pid_to_lh_to_pos_rank[pid].items()) for pid in sorted(self.pid_to_lh_to_pos_rank)])
        return hash((str_1, str_2))

    def __str__(self):
        rank_to_labels = defaultdict(set)
        for pid, lh_to_rank in self.pid_to_lh_to_pos_rank.items():
            if 'L' in lh_to_rank:
                rank_to_labels[lh_to_rank['L']].add(f'{pid}B')
            if 'H' in lh_to_rank:
                rank_to_labels[lh_to_rank['H']].add(f'{pid}A')

        pos, labels = [], []
        for rank, sorted_labels in sorted(rank_to_labels.items()):
            pos.append(str(self.positions[rank]))
            labels.append('+'.join(sorted_labels))

        return f"(pos)[{', '.join(pos)}] (label)[{', '.join(labels)}]"

    def calculate_insertion_positions_iterator(self, step, item, inferencer: TwoLabelInferencer):

        low, high = -1, len(self.positions)
        if item in inferencer.item_to_pid_and_contributing_lh:
            for pid, contributing_lh in inferencer.item_to_pid_and_contributing_lh[item]:
                if contributing_lh == 'L' and self.is_tracking_pid_lh(pid, 'H'):
                    high = min(high, self.pid_to_lh_to_pos_rank[pid]['H'])
                elif contributing_lh == 'H' and self.is_tracking_pid_lh(pid, 'L'):
                    low = max(low, self.pid_to_lh_to_pos_rank[pid]['L'])

        for pid in inferencer.sharing_item_to_pids.get(item, set()):
            if self.is_tracking_pid_lh(pid, 'H'):
                high = min(high, self.pid_to_lh_to_pos_rank[pid]['H'])
            if self.is_tracking_pid_lh(pid, 'L'):
                low = max(low, self.pid_to_lh_to_pos_rank[pid]['L'])

        if low < 0:
            low_bound_pos = 0
        else:
            low_bound_pos = self.positions[low] + 1

        if high < len(self.positions):
            high_bound_pos = self.positions[high]
        else:
            high_bound_pos = step

        return range(low_bound_pos, high_bound_pos + 1)

    def insert(self, step, item, position, inferencer: TwoLabelInferencer):

        state = deepcopy(self)

        # index of insertion position in state.positions
        pos_idx = bisect_left(self.positions, position)

        # increase position values by 1 accordingly
        for idx in range(pos_idx, len(self.positions)):
            state.positions[idx] += 1

        is_an_item_in_pattern = item in inferencer.item_to_pid_and_contributing_lh
        is_a_sharing_item = item in inferencer.sharing_item_to_pids

        if is_an_item_in_pattern or is_a_sharing_item:

            # insert new pos into state.positions
            state.positions.insert(pos_idx, position)

            # increase rank values by 1 accordingly
            for pid, tracking_ranks in self.pid_to_lh_to_pos_rank.items():
                if 'L' in tracking_ranks and tracking_ranks['L'] >= pos_idx:
                    state.pid_to_lh_to_pos_rank[pid]['L'] += 1

                    if 'H' in tracking_ranks:
                        state.pid_to_lh_to_pos_rank[pid]['H'] += 1

                elif 'H' in tracking_ranks and tracking_ranks['H'] >= pos_idx:
                    state.pid_to_lh_to_pos_rank[pid]['H'] += 1

            # re-calculate representative label ranks
            if is_an_item_in_pattern:
                for pid, contributing_lh in inferencer.item_to_pid_and_contributing_lh[item]:

                    if contributing_lh == 'L':
                        if state.is_tracking_pid_lh(pid, 'L'):
                            current_l = state.pid_to_lh_to_pos_rank[pid]['L']
                            state.pid_to_lh_to_pos_rank[pid]['L'] = max(current_l, pos_idx)
                        else:
                            state.pid_to_lh_to_pos_rank.setdefault(pid, {})['L'] = pos_idx
                    else:
                        if state.is_tracking_pid_lh(pid, 'H'):
                            current_h = state.pid_to_lh_to_pos_rank[pid]['H']
                            state.pid_to_lh_to_pos_rank[pid]['H'] = min(current_h, pos_idx)
                        else:
                            state.pid_to_lh_to_pos_rank.setdefault(pid, {})['H'] = pos_idx

            elif is_a_sharing_item:
                for pid in inferencer.sharing_item_to_pids[item]:
                    if state.is_tracking_pid_lh(pid, 'L'):
                        current_l = state.pid_to_lh_to_pos_rank[pid]['L']
                        state.pid_to_lh_to_pos_rank[pid]['L'] = max(current_l, pos_idx)
                    else:
                        state.pid_to_lh_to_pos_rank.setdefault(pid, {})['L'] = pos_idx

                    if state.is_tracking_pid_lh(pid, 'H'):
                        current_h = state.pid_to_lh_to_pos_rank[pid]['H']
                        state.pid_to_lh_to_pos_rank[pid]['H'] = max(current_h, pos_idx)
                    else:
                        state.pid_to_lh_to_pos_rank.setdefault(pid, {})['H'] = pos_idx

            if step in inferencer.step_to_pids_to_stop_tracking_lhs:
                for pid, boundaries in inferencer.step_to_pids_to_stop_tracking_lhs[step].items():
                    if 'L' in boundaries and state.is_tracking_pid_lh(pid, 'L'):
                        del state.pid_to_lh_to_pos_rank[pid]['L']
                    if 'H' in boundaries and state.is_tracking_pid_lh(pid, 'H'):
                        del state.pid_to_lh_to_pos_rank[pid]['H']
                    if not state.pid_to_lh_to_pos_rank[pid]:
                        del state.pid_to_lh_to_pos_rank[pid]
            state.compact()

        return state

    def compact(self):
        """
        remove positions that no label is tracking.
        """
        valid_ranks = set()
        for tracking_ranks in self.pid_to_lh_to_pos_rank.values():
            valid_ranks.update(tracking_ranks.values())

        missing_ranks = [rank for rank in range(len(self.positions) - 1, -1, -1) if rank not in valid_ranks]

        for missing_rank in missing_ranks:

            self.positions.remove(self.positions[missing_rank])

            for pid, tracking_ranks in self.pid_to_lh_to_pos_rank.items():
                if 'L' in tracking_ranks and tracking_ranks['L'] > missing_rank:
                    self.pid_to_lh_to_pos_rank[pid]['L'] -= 1

                    if 'H' in tracking_ranks:
                        self.pid_to_lh_to_pos_rank[pid]['H'] -= 1

                elif 'H' in tracking_ranks and tracking_ranks['H'] > missing_rank:
                    self.pid_to_lh_to_pos_rank[pid]['H'] -= 1

    def is_tracking_pid_lh(self, pid, lh):
        return pid in self.pid_to_lh_to_pos_rank and lh in self.pid_to_lh_to_pos_rank[pid]

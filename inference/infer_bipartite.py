from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
from multiprocessing.pool import Pool
from os import cpu_count
from time import time
from typing import List, Tuple, Dict, Set

import networkx as nx

from core.mallows import Mallows
from core.patterns import BipartitePattern


class BipartiteInferencer(object):
    """It performs inference for a list of non-transitive patterns over a Mallows model, i.e., calculating the probability
    that a random sampled permutation satisfies at least one pattern.
    """

    def __init__(self, mallows: Mallows):

        self.mallows: Mallows = mallows
        self.patterns: List[BipartitePattern] = None
        self.item_to_pids_to_labels: Dict[object, Dict[int, Set]] = None

    def calculate_item_to_pids_to_labels(self):
        item_to_pids_to_labels = {}
        for pid, pattern in enumerate(self.patterns):
            for label, items in pattern.label_to_items.items():
                for item in items:
                    item_to_pids_to_labels.setdefault(item, {}).setdefault(pid, set()).add(label)

        return item_to_pids_to_labels

    def solve(self, patterns: List[BipartitePattern], threads=None) -> Tuple[float, int]:
        start_time = time()  # time in seconds
        threads = threads or cpu_count()
        step_max = self.mallows.m - 1

        self.patterns = deepcopy(patterns)
        for pattern in self.patterns:
            pattern.remove_edges_sharing_multiple_items()  # prune edges if two labels share at least 2 items.
            pattern.calculate_finishing_steps(self.mallows.center)

        self.item_to_pids_to_labels = self.calculate_item_to_pids_to_labels()

        ustate_to_prob: Dict[State, float] = defaultdict(float)
        prob_accumulated = 0.0

        for step, item in enumerate(self.mallows.center):
            print(f'step {step} / {step_max}, #states = {len(ustate_to_prob)}, prob_now = {prob_accumulated}')

            if step == 0:
                init_state = State(self.patterns)
                init_state = init_state.insert(step, item, 0, self)
                ustate_to_prob[init_state] = 1

            elif ustate_to_prob:
                tasks = list(ustate_to_prob.items())
                pij_i = self.mallows.pij_matrix[step]
                batched_tasks = [(tasks[i::threads], step, item, pij_i) for i in range(threads)]

                with Pool(processes=threads) as pool:
                    res_list = pool.map(self.perform_a_batch_of_tasks, batched_tasks)

                ustate_to_prob.clear()
                for res, prob_satisfied in res_list:
                    prob_accumulated += prob_satisfied
                    for state, probs in res.items():
                        ustate_to_prob[state] += sum(probs)

        runtime_ms = int(1000 * (time() - start_time))
        return prob_accumulated, runtime_ms

    def perform_a_batch_of_tasks(self, task_batch):

        state_and_prob_batch, step, item, pij_i = task_batch

        state_to_probs = defaultdict(list)
        prob_satisfied = 0

        for state, prob in state_and_prob_batch:
            for position in range(step + 1):
                state_new = state.insert(step, item, position, self)
                if state_new.is_satisfying:
                    prob_satisfied += prob * pij_i[position]
                elif not state_new.is_violating:
                    additive_prob = prob * pij_i[position]
                    state_to_probs[state_new].append(additive_prob)

        return state_to_probs, prob_satisfied


class State(object):
    """This is a state in Dynamic Programming for star query evaluation.

    Given an item, I want to know
        1) which patterns and labels this item belongs to.
        2) are these labels preferred labels.

    After inserting an item at a position,
        See whether related edges are violated or satisfied, and update accordingly.
    """

    def __init__(self, patterns: List[BipartitePattern]):

        self.positions = []
        self.pid_to_labels_to_rank = {}
        self.upid_to_ugraph: Dict[int, nx.DiGraph] = {pid: deepcopy(patt.graph) for pid, patt in enumerate(patterns)}

        self.is_satisfying = False
        self.is_violating = False

    def __eq__(self, other):
        return isinstance(other, State) and \
               (self.positions == other.positions) and \
               (self.pid_to_labels_to_rank == other.pid_to_labels_to_rank) and \
               (self.upid_to_ugraph.keys() == other.upid_to_ugraph.keys()) and \
               all([self.upid_to_ugraph[pid].edges == other.upid_to_ugraph[pid].edges for pid in
                    self.upid_to_ugraph]) and \
               (self.is_satisfying == other.is_satisfying) and \
               (self.is_violating == other.is_violating)

    def __hash__(self):
        str_1 = str(self.positions)
        str_2 = str([(pid, sorted(self.pid_to_labels_to_rank[pid])) for pid in sorted(self.pid_to_labels_to_rank)])
        str_3 = str([(pid, sorted(self.upid_to_ugraph[pid].edges)) for pid in sorted(self.upid_to_ugraph)])
        return hash((str_1, str_2, str_3, self.is_satisfying, self.is_violating))

    def __str__(self):
        rank_to_labels = {}
        for pid, label_to_rank_dict in self.pid_to_labels_to_rank.items():
            for label, rank in label_to_rank_dict.items():
                rank_to_labels.setdefault(rank, set()).add((pid, label))

        pos, labels = [], []
        for rank, tuples in sorted(rank_to_labels.items()):
            pos.append(str(self.positions[rank]))
            sorted_labels = [f'{pid}-{label}' for pid, label in sorted(tuples)]
            labels.append('+'.join(sorted_labels))

        return f"(pos)[{', '.join(pos)}] (label)[{', '.join(labels)}]"

    def insert(self, step: int, item, position: int, inferencer: BipartiteInferencer):
        """Return a new state instance by inserting an item into current state."""
        state = deepcopy(self)
        pos_idx = bisect_left(self.positions, position)  # index of insertion position in state.positions

        if item in inferencer.item_to_pids_to_labels:
            # update position values
            state.positions_right_shift_by_1_since_index(pos_idx)
            # update rank values
            state.positions_add_new_pos_at_index(position, pos_idx)
            # re-calculate representative items for labels, considering current item is being inserted
            state.update_tracking_positions(item, pos_idx, inferencer)
            # update uncertain pids and edges
            state.update_uncertain_graphs(step, inferencer)

            if state.is_satisfying or state.is_violating:
                return state
            else:
                state.compact()

        # Current item is not related to any pattern, then just right shift values in state.positions.
        else:
            state.positions_right_shift_by_1_since_index(pos_idx)

        return state

    def is_tracking(self, pid, label):
        return pid in self.pid_to_labels_to_rank and label in self.pid_to_labels_to_rank[pid]

    def is_tracking_two_labels_in_one_pid(self, pid, label_1, label_2):
        return self.is_tracking(pid, label_1) and label_2 in self.pid_to_labels_to_rank[pid]

    def positions_right_shift_by_1_since_index(self, idx):
        """Right shift some self.positions values by 1 due to new item insertion."""
        for idx in range(idx, len(self.positions)):
            self.positions[idx] += 1

    def positions_add_new_pos_at_index(self, pos, idx):
        """When inserting new pos into self.positions, pid_to_label_to_rank is also updated."""
        self.positions.insert(idx, pos)

        # TODO optimization
        for pid, label_to_rank in self.pid_to_labels_to_rank.items():
            for label, rank in label_to_rank.items():
                if rank >= idx:
                    self.pid_to_labels_to_rank[pid][label] += 1

    def update_tracking_positions(self, item, pos_idx, inferencer: BipartiteInferencer):
        """recalculate representative item for (pid, label)"""

        for pid in set(inferencer.item_to_pids_to_labels[item]).intersection(set(self.upid_to_ugraph)):
            pattern = inferencer.patterns[pid]
            labels = set(inferencer.item_to_pids_to_labels[item][pid])
            labels.intersection_update(set(self.upid_to_ugraph[pid].nodes))
            for label in labels:
                if self.is_tracking(pid, label):
                    rank_tracking = self.pid_to_labels_to_rank[pid][label]
                    if pattern.is_l_type(label) != (pos_idx > rank_tracking):
                        self.pid_to_labels_to_rank[pid][label] = pos_idx
                else:
                    self.pid_to_labels_to_rank.setdefault(pid, {})[label] = pos_idx

    def edge_satisfied(self, pid, parent, child):
        graph: nx.DiGraph = self.upid_to_ugraph[pid]
        graph.remove_edge(parent, child)

        if not nx.degree(graph, parent):
            graph.remove_node(parent)

        if not nx.degree(graph, child):
            graph.remove_node(child)

        # if all edges of pid are satisfied
        if not graph:
            self.is_satisfying = True

    def pid_violated(self, pid):
        del self.pid_to_labels_to_rank[pid]
        del self.upid_to_ugraph[pid]

        # if all patterns are violated
        if not self.upid_to_ugraph:
            self.is_violating = True

    def update_uncertain_graphs(self, step, inferencer: BipartiteInferencer):

        satisfied_edges, violated_pids = {}, set()
        for pid, graph in self.upid_to_ugraph.items():
            for parent, child in graph.edges:
                if self.is_tracking_two_labels_in_one_pid(pid, parent, child):
                    parent_pos = self.pid_to_labels_to_rank[pid][parent]
                    child_pos = self.pid_to_labels_to_rank[pid][child]
                    # an edge is satisfied
                    if parent_pos < child_pos:
                        satisfied_edges.setdefault(pid, set()).add((parent, child))
                    else:
                        parent_finished = (step >= inferencer.patterns[pid].label_to_finishing_step[parent])
                        child_finished = (step >= inferencer.patterns[pid].label_to_finishing_step[child])
                        # if current pid is already violated
                        if parent_finished and child_finished:
                            violated_pids.add(pid)
                            break

        for pid in violated_pids:
            self.pid_violated(pid)
            if pid in satisfied_edges:
                del satisfied_edges[pid]

        if not self.is_violating:
            for pid, edges in satisfied_edges.items():
                for (parent, child) in edges:
                    self.edge_satisfied(pid, parent, child)

                if self.is_satisfying:
                    break

    def compact(self):
        """Remove positions that no label is tracking."""
        valid_ranks = set()
        for label_to_rank_dict in self.pid_to_labels_to_rank.values():
            valid_ranks.update(label_to_rank_dict.values())

        missing_ranks = [rank for rank in range(len(self.positions) - 1, -1, -1) if rank not in valid_ranks]

        for missing_rank in missing_ranks:

            self.positions.remove(self.positions[missing_rank])

            for pid, label_to_rank_dict in self.pid_to_labels_to_rank.items():
                for label, rank in label_to_rank_dict.items():
                    if rank > missing_rank:
                        self.pid_to_labels_to_rank[pid][label] -= 1

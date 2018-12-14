from bisect import bisect_right
from copy import deepcopy
from random import random
from typing import Set, Dict

import networkx as nx
from numpy.random import choice
from scipy.misc import comb

GRAPH_ITEMS_SEP = '--'
PATTERN_SEP = ' <> '


class Pattern(object):

    def __init__(self, label_to_children: Dict[object, Set] = None, label_to_items: Dict[object, Set] = None):
        """
        A general pattern consists of (1) a graph of labels; (2) items in each label.

        Note: there is no Label class yet. Labels are just names, typically strings.
        """
        self.label_to_children = deepcopy(label_to_children or {})
        self.label_to_items = deepcopy(label_to_items or {})

        self.graph = nx.DiGraph()
        for node, children in self.label_to_children.items():
            for child in children:
                self.graph.add_edge(node, child)

        self.tc: nx.DiGraph = None
        self.label_to_finishing_step: Dict[object, int] = None
        self.step_to_finishing_labels: Dict[int, Set] = None

        self.order_to_label_to_finishing_step: Dict[str, Dict[object, int]] = {}
        self.order_to_step_to_finishing_labels: Dict[str, Dict[int, Set]] = {}

    def __eq__(self, other):
        return isinstance(other, Pattern) and \
               (self.label_to_items == other.label_to_items) and \
               (self.label_to_children == other.label_to_children)

    def __hash__(self):
        str_1 = str([(label, sorted(self.label_to_items[label])) for label in sorted(self.label_to_items)])
        str_2 = str(sorted(self.graph.edges))
        return hash((str_1, str_2))

    def __str__(self):
        return f'{self.label_to_children}{GRAPH_ITEMS_SEP}{self.label_to_items}'

    def to_string(self):
        return f"""
                pattern:   {self.label_to_children}
                items:     {self.label_to_items}
                """

    def iter_direct_children_of_label(self, label):
        return self.graph.successors(label)

    def iter_direct_parents_of_label(self, label):
        return self.graph.predecessors(label)

    def get_items_in_label(self, label):
        return self.label_to_items[label]

    def calculate_tc(self):
        self.tc = nx.transitive_closure(self.graph)

    def get_all_ancestor_labels(self, label):
        if not self.tc:
            self.calculate_tc()

        if self.tc.has_node(label):
            return self.tc.predecessors(label)
        else:
            return set()

    def get_all_descendant_labels(self, label):
        if not self.tc:
            self.calculate_tc()

        if self.tc.has_node(label):
            return self.tc.successors(label)
        else:
            return set()

    def calculate_finishing_steps(self, insertion_order):
        self.label_to_finishing_step = {}
        for label, items in self.label_to_items.items():
            steps = [insertion_order.index(item) for item in items]
            self.label_to_finishing_step[label] = max(steps)

        self.step_to_finishing_labels = {}
        for label, step in self.label_to_finishing_step.items():
            self.step_to_finishing_labels.setdefault(step, set()).add(label)

    def calculate_finishing_steps_given_list(self, insertion_orders):
        for insertion_order in insertion_orders:
            order = str(insertion_order)
            self.order_to_label_to_finishing_step[order] = {}
            for label, items in self.label_to_items.items():
                steps = [insertion_order.index(item) for item in items]
                self.order_to_label_to_finishing_step[order][label] = max(steps)

            self.order_to_step_to_finishing_labels[order] = {}
            for label, step in self.order_to_label_to_finishing_step[order].items():
                self.order_to_step_to_finishing_labels[order].setdefault(step, set()).add(label)

    def calculate_sub_tc(self, subset_labels: Set):
        sub_pattern = Pattern({}, {})
        sub_pattern.label_to_items = {label: self.label_to_items[label] for label in subset_labels}
        sub_pattern.graph = nx.DiGraph(self.tc.subgraph(subset_labels))
        sub_pattern.tc = sub_pattern.graph
        return sub_pattern

    def calculate_label_to_top_matching_item_rank(self, ranking):

        # item to its rank in ranking
        item_to_rank = {item: rank for rank, item in enumerate(ranking)}

        # label to the ranks of its items in ranking
        label_to_sorted_ranks = {}
        for label, items in self.label_to_items.items():
            label_to_sorted_ranks[label] = sorted([item_to_rank[item] for item in items])

        label_to_match_rank = {}
        for label in nx.topological_sort(self.graph):
            parent_ranks = [label_to_match_rank[parent] for parent in self.graph.predecessors(label)]
            latest_parent_rank = max(parent_ranks, default=-1)
            matching_idx = bisect_right(label_to_sorted_ranks[label], latest_parent_rank)
            if matching_idx < len(label_to_sorted_ranks[label]):
                label_to_match_rank[label] = label_to_sorted_ranks[label][matching_idx]
            else:
                return False, {}

        return True, label_to_match_rank

    def is_satisfying_ranking(self, ranking):
        return self.calculate_label_to_top_matching_item_rank(ranking)[0]

    def estimate_num_states_generated_during_ltm(self, insertion_order):
        """
        [0, a, b, c] are 4 span values for a label, where
            - [a, b] is the span of items in current label,
            - [0, c] is the life span considering its neighbors.
        """

        label_to_parents = {}
        for parent, children in self.label_to_children.items():
            for child in children:
                label_to_parents.setdefault(child, set()).add(parent)

        label_to_span = {}
        for idx, item in enumerate(insertion_order):
            for label, items in self.label_to_items.items():
                if item in items:
                    span = label_to_span.get(label, [len(insertion_order), 0])
                    label_to_span[label] = [min(idx, span[0]), max(idx, span[1])]

        label_to_life = {}
        for label in self.label_to_items:
            life = label_to_span[label][1]

            for child in self.label_to_children.get(label, set()):
                life = max(life, label_to_span[child][1])

            for parent in label_to_parents.get(label, set()):
                life = max(life, label_to_span[parent][1])

            label_to_life[label] = life

        max_step = max(label_to_life.values())

        cost = 0
        for step in range(max_step):
            num_q = len(label_to_span)  # #buckets to track
            num_w = 0  # #buckets that are either empty or filled
            for label in label_to_span:
                span = label_to_span[label]
                if span[0] <= step < span[1]:
                    num_w += 1

            cost_at_step = int(comb(step + num_q, num_q)) * (2 ** num_w)
            cost += cost_at_step

        return cost

    def is_acyclic(self):
        return nx.is_directed_acyclic_graph(self.graph)

    @classmethod
    def from_string(cls, pattern_str):
        graph, items = pattern_str.split(GRAPH_ITEMS_SEP)
        return Pattern(eval(graph), eval(items))


class BipartitePattern(Pattern):
    """A bipartite pattern has L-type labels and R-type labels."""

    def __init__(self, label_to_children, label_to_items):
        super().__init__(label_to_children, label_to_items)

        self.l_type_labels: Set = set()
        for label in self.graph:
            if next(self.graph.successors(label), None):
                self.l_type_labels.add(label)

    def is_l_type(self, label):
        return label in self.l_type_labels

    def remove_edges_sharing_multiple_items(self):

        for parent, children in deepcopy(self.label_to_children).items():
            parent_items = self.label_to_items[parent]
            for child in children:
                if len(parent_items.intersection(self.label_to_items[child])) > 1:
                    print(f'removing ({parent}, {child})')
                    self.label_to_children[parent].remove(child)
                    self.graph.remove_edge(parent, child)

        for node in set(self.graph.nodes()):
            if nx.degree(self.graph, node) == 0:
                self.graph.remove_node(node)
                del self.label_to_items[node]
                if node in self.label_to_children:
                    del self.label_to_children[node]
                if node in self.l_type_labels:
                    self.l_type_labels.remove(node)

        self.tc = None
        self.label_to_finishing_step = None
        self.step_to_finishing_labels = None

    @classmethod
    def from_string(cls, pattern_str):
        graph, items = pattern_str.split(GRAPH_ITEMS_SEP)
        return BipartitePattern(eval(graph), eval(items))


class TwoLabelPattern(BipartitePattern):

    def __init__(self, label_to_children, label_to_items):
        super().__init__(label_to_children, label_to_items)

        self.better_label = next(iter(label_to_children))
        self.worse_label = next(iter(label_to_children[self.better_label]))

    def is_better_label(self, label):
        return self.better_label == label

    @classmethod
    def from_string(cls, pattern_str):
        graph, items = pattern_str.split(GRAPH_ITEMS_SEP)
        return TwoLabelPattern(eval(graph), eval(items))


class ItemPref(Pattern):
    """
    ItemPattern is essentially item-level preferences.
    """

    def __init__(self, parent_to_children):
        items = set(parent_to_children)
        for children in parent_to_children.values():
            items.update(set(children))

        label_to_children = dict(parent_to_children)
        label_to_items = {item: {item} for item in items}

        super().__init__(label_to_children, label_to_items)

    def __str__(self):
        return str(self.label_to_children)

    def __eq__(self, other):
        return isinstance(other, ItemPref) and (self.label_to_children == other.label_to_children)

    def __hash__(self):
        return hash(str(sorted(self.graph.edges)))

    def encode(self):
        encoded = [[parent, sorted(self.label_to_children[parent])] for parent in sorted(self.label_to_children)]
        return str(encoded)

    def is_a_linear_extension(self, permutation):
        item_to_rank = {item: rank for rank, item in enumerate(permutation)}
        return self.is_a_linear_extension_by_item_ranks(item_to_rank)

    def is_a_linear_extension_by_item_ranks(self, item_to_rank):
        for parent, children in self.label_to_children.items():
            parent_rank = item_to_rank[parent]
            for child in children:
                if item_to_rank[child] < parent_rank:
                    return False

        return True

    @classmethod
    def from_string(cls, pref_str):
        return ItemPref(eval(pref_str))

    @classmethod
    def generate_random_pref(cls, m=20, num_items=5, edge_prob=0.3):
        items = choice(m, num_items, False)
        pref = {}
        for i in range(num_items - 1):
            for j in range(i + 1, num_items):
                if random() < edge_prob:
                    pref.setdefault(items[i], set()).add(items[j])

        if pref:
            return ItemPref(pref)
        else:
            return cls.generate_random_pref(m, num_items, edge_prob)


if __name__ == '__main__':

    from core.mallows import Mallows
    from experiment_code.utils import generate_random_label_to_children

    m = 15
    phi = 0.2

    model = Mallows(list(range(m)), phi)

    for _ in range(1000):
        ranking = model.sample_a_permutation()

        pref = ItemPref(generate_random_label_to_children(model.center))

        res1 = pref.is_satisfying_ranking(ranking)
        res2 = pref.is_a_linear_extension(ranking)

        if res1 != res2:
            print('WRONG!')
            print(pref)
            print(ranking)
            print(res1, res2)
            print(pref.calculate_label_to_top_matching_item_rank(ranking))
            print(pref.label_to_items)
            print(pref.label_to_children)
            break

        print(res1)

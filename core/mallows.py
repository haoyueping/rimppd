from math import factorial
from random import choices
from typing import List, Tuple

from core.patterns import ItemPref
from helpers.utils import normalize_numeric_list, calculate_kendall_tau_distance


class Mallows(object):

    def __init__(self, center: List, phi: float):
        self.center = list(center)
        self.phi = phi

        self.m: int = len(self.center)
        self.item_to_rank = {item: rank for rank, item in enumerate(self.center)}

        self.pij_matrix: Tuple[Tuple[float]] = self.calculate_pij_matrix()
        self.normalization_constant = self.calculate_normalization_constant()

    def __str__(self):
        return f'Mallows(center={self.center}, phi={self.phi})'

    def get_prob_i_j(self, i, j) -> float:
        return self.pij_matrix[i][j]

    def get_rank_of_item(self, item):
        return self.item_to_rank[item]

    def sample_a_ranking(self) -> List:
        ranking = []
        insertion_range = []

        for step, item in enumerate(self.center):
            insertion_range.append(step)
            sample_index = choices(insertion_range, weights=self.pij_matrix[step])[0]

            ranking.insert(sample_index, item)

        return ranking

    def sample_a_permutation(self) -> List:
        return self.sample_a_ranking()

    def calculate_normalization_constant(self) -> float:
        try:
            norm = (1 - self.phi) ** (-self.m)
            for i in range(1, self.m + 1):
                norm *= (1 - self.phi ** i)
        except ZeroDivisionError:
            norm = factorial(self.m)
        return norm

    def calculate_kendall_tau_distance(self, permutation) -> int:
        return calculate_kendall_tau_distance(self.center, permutation)

    def calculate_prob_by_distance(self, distance):
        return (self.phi ** distance) / self.normalization_constant

    def calculate_prob_of_permutation(self, permutation):
        dist = self.calculate_kendall_tau_distance(permutation)
        return self.calculate_prob_by_distance(dist)

    def calculate_kendall_tau_distance_of_item_pref(self, pref: ItemPref):
        dist = 0
        label_to_center_rank = {label: self.get_rank_of_item(label) for label in pref.label_to_items}
        for parent, children in pref.label_to_children.items():
            parent_rank = label_to_center_rank[parent]
            for child in children:
                if parent_rank > label_to_center_rank[child]:
                    dist += 1
        return dist

    def calculate_pij_matrix(self):

        pij = []
        for i in range(self.m):
            pi = [self.phi ** (i - j) for j in range(i + 1)]
            pi = normalize_numeric_list(pi)
            pij.append(tuple(pi))

        return tuple(pij)


if __name__ == '__main__':
    import random

    center = list(range(10))
    random.shuffle(center)

    mallows = Mallows(center, 0.5)
    print(mallows)
    print(mallows.calculate_normalization_constant())

    for _ in range(5):
        r = mallows.sample_a_permutation()
        pref = ItemPref.generate_random_pref(5, 4)
        print(mallows.calculate_kendall_tau_distance(r), calculate_kendall_tau_distance(mallows.center, r), ' || ', r)
        print(mallows.calculate_kendall_tau_distance_of_item_pref(pref), pref)

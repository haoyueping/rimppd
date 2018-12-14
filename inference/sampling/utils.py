from math import e

import pandas as pd

from core.mallows import Mallows
from core.patterns import ItemPref, Pattern, PATTERN_SEP


def get_itempref_of_2_succ_3_succ_m(m=10) -> ItemPref:
    return ItemPref({i: {i + 1} for i in range(1, m - 1)})


def get_test_case_of_itempref(pid=0):
    row = pd.read_csv('data/test_cases_item_prefs.csv').iloc[pid]
    pref = ItemPref.from_string(row['pref'])
    mallows = Mallows(list(range(row['m'])), row['phi'])
    p_exact = e ** row['log_p']
    return pref, mallows, p_exact


def get_test_case_of_pattern(pid=0):
    row = pd.read_csv('data/test_cases_label_patterns.csv').iloc[pid]
    pattern = Pattern.from_string(row['pattern'])
    mallows = Mallows(list(range(row['m'])), row['phi'])
    p_exact = e ** row['log_p']
    return pattern, mallows, p_exact


def get_test_case_of_patterns_from_movielens_2_labels(rid=0):
    p_exact = pd.read_csv('data/output_movielens_ramp-vs-amp_2labels_exact.csv').loc[rid, 'p_exact']

    row = pd.read_csv('data/input_movielens_ramp-vs-amp_2labels.csv').loc[rid]
    center = eval(row['ranking'])
    mallows = Mallows(center=center, phi=row['phi'])
    patterns = [Pattern.from_string(pattern_str) for pattern_str in row['patterns'].split(PATTERN_SEP)]

    return patterns, mallows, p_exact


def get_test_case_of_patterns_from_movielens_linear(rid=0):
    row = pd.read_csv('data/input_movielens_ramp-vs-amp.csv').loc[rid]
    center = eval(row['ranking'])
    mallows = Mallows(center=center, phi=row['phi'])
    patterns = [Pattern.from_string(pattern_str) for pattern_str in row['patterns'].split(PATTERN_SEP)]

    return patterns, mallows


def get_test_case_of_patterns_from_movielens_5_labels(rid=0):
    """
    Hard cases for rAMP are 36, 52, 68, 84, 100, 116, 132, 148
    """
    row = pd.read_csv('data/input_movielens_ramp-vs-amp_5_labels.csv').loc[rid]

    mallows = Mallows(center=eval(row['ranking']), phi=row['phi'])
    patterns = [Pattern.from_string(pattern_str) for pattern_str in row['patterns'].split(' <> ')]

    return patterns, mallows


def get_test_case_of_patterns_from_synthetic_4_labels(pid=0):
    df_ans = pd.read_csv('data/test_cases_4_labels_sharing_BD_3_subs_convergence_by_ramp_3.csv')
    df_ans = df_ans.groupby('rid').first()
    p_exact = df_ans.loc[pid, 'p_exact']

    row = pd.read_csv('data/test_cases_4_labels_sharing_BD_3_subs.csv').loc[pid]
    patterns_str = row['pref(A>C|A>D|B>D)']
    patterns = [Pattern.from_string(pattern_str) for pattern_str in patterns_str.split('\n')]
    mallows = Mallows(list(range(row['m'])), row['phi'])
    return patterns, mallows, p_exact


if __name__ == '__main__':
    res = get_test_case_of_patterns_from_movielens_5_labels()
    print(res)

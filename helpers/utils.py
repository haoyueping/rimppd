import smtplib
from datetime import datetime
from math import sqrt
from typing import List, Dict, Set

import numpy as np


def send_email(subject=None, body=None):
    subject = subject or 'Running experiment'

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = body or f'Experiment result at {current_time}'

    s_from = 'pymail4exp@gmail.com'
    s_to = 'timoping@gmail.com'

    email_text = f'From: {s_from}\nTo: {s_to}\nSubject: {subject}\n\n{body}'

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.ehlo()
        server.login(s_from, 'pYmail--4--exp')
        server.sendmail(s_from, s_to, email_text)
        server.close()

        print('Email sent!')


def normalize_numeric_list(numbers: List):
    return [num / sum(numbers) for num in numbers]


def topological_sort_reversed_recursive(nodes: Set, node_to_children: Dict[object, Set]):
    def _dfs_util(node):
        parents.add(node)

        for w in node_to_children.get(node, set()):

            if w not in seen:
                _dfs_util(w)

        parents.remove(node)
        seen.add(node)
        order.append(node)

    parents = set()
    seen = set()
    order = []

    for v in nodes:
        if v not in seen:
            _dfs_util(v)

    return order


def calculate_rsd_by_square_sum_and_avg(prob_x_square_sum, num_samples, avg):
    variance = prob_x_square_sum / num_samples - (avg * avg)
    return sqrt(variance) / avg


def calculate_kendall_tau_distance(x, y):
    """Code from Scipy. See details in
    https://github.com/scipy/scipy/blob/v0.15.1/scipy/stats/stats.py#L2827

    """

    x = np.asarray(x)
    y = np.asarray(y)

    n = np.int64(len(x))
    temp = list(range(n))  # support structure used by mergesort

    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support
    # returns the number of swaps required by an equivalent bubble sort

    def mergesort(offs, length):
        exchcnt = 0
        if length == 1:
            return 0
        if length == 2:
            if y[perm[offs]] <= y[perm[offs + 1]]:
                return 0
            t = perm[offs]
            perm[offs] = perm[offs + 1]
            perm[offs + 1] = t
            return 1
        length0 = length // 2
        length1 = length - length0
        middle = offs + length0
        exchcnt += mergesort(offs, length0)
        exchcnt += mergesort(middle, length1)
        if y[perm[middle - 1]] < y[perm[middle]]:
            return exchcnt
        # merging
        i = j = k = 0
        while j < length0 or k < length1:
            if k >= length1 or (j < length0 and y[perm[offs + j]] <=
                                y[perm[middle + k]]):
                temp[i] = perm[offs + j]
                d = i - j
                j += 1
            else:
                temp[i] = perm[middle + k]
                d = (offs + i) - (middle + k)
                k += 1
            if d > 0:
                exchcnt += d
            i += 1
        perm[offs:offs + length] = temp[0:length]
        return exchcnt

    # initial sort on values of x and, if tied, on values of y
    perm = np.lexsort((y, x))

    # count exchanges
    exchanges = mergesort(0, n)

    return exchanges


if __name__ == '__main__':
    graph = {'D': {'B', 'C'}, 'A': {'B'}}
    print(topological_sort_reversed_recursive({'A', 'B', 'C', 'D'}, graph))

from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from time import time
from typing import Tuple

from core.mallows import Mallows
from core.patterns import ItemPref


def worker(paras: Tuple[ItemPref, Mallows, int]):
    itempref, mallows, single_core_workload = paras
    count = 0
    for _ in range(single_core_workload):
        permutation = mallows.sample_a_permutation()
        if itempref.is_a_linear_extension(permutation):
            count += 1

    return count


def estimate_itempref_by_rejection(itempref: ItemPref, mallows: Mallows, num_samples=1000):
    num_cpu = cpu_count()

    single_core_workload = num_samples // num_cpu
    num_samples = single_core_workload * num_cpu

    start = time()
    with Pool() as pool:
        count_list = pool.map(worker, [(itempref, mallows, single_core_workload) for _ in range(num_cpu)])

    runtime = int(1000 * (time() - start))
    return sum(count_list) / num_samples, runtime

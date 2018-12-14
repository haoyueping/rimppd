import json
import os
import subprocess
from platform import system as get_os_name

from core.mallows import Mallows
from core.patterns import Pattern


def calculate_jvm_xmx():
    """
    Calculate JVM heap size for ltm.jar execution.
    """
    os_name = get_os_name()

    if os_name == 'Linux':
        res = subprocess.run(['free', '-g'], stdout=subprocess.PIPE)
        res = res.stdout.decode('utf-8')
        size_gb = int(res.split('Mem:')[1].split()[0])

    # Darwin represents macOS here.
    elif os_name == 'Darwin':
        res = subprocess.run(['sysctl', 'hw.memsize'], stdout=subprocess.PIPE)
        res = res.stdout.decode('utf-8')
        size_gb = int(res.split(': ')[1]) // (2 ** 30)

    # Other platforms, e.g., Windows.
    else:
        size_gb = 8  # no optimization for other platforms yet.

    return size_gb - 2


def calculate_marginal_prob_over_mallows_by_ltm(pattern: Pattern, mallows: Mallows, num_cores=None, timeout=None):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    jar_file = cur_dir + '/ltm.jar'
    json_file = cur_dir + '/temp.json'
    ltm_verbose_file = cur_dir + '/ltm_verbose.txt'

    # num_cores is for ltm.jar multi-threading
    num_cores = num_cores or os.cpu_count()

    # represent a pattern by a list of nodes. Each node has three properties {name, items, children}.
    nodes = []
    for node_name, items in pattern.label_to_items.items():
        items_in_node = [int(item) for item in items]
        children_names = [f'L-{child}' for child in pattern.iter_direct_children_of_label(node_name)]
        nodes.append({'name': f'L-{node_name}', 'items': items_in_node, 'children': children_names})
    # save pattern and Mallows core info in a local JSON file
    print(nodes)
    with open(json_file, 'w') as outfile:
        json.dump({'pattern': nodes, 'center': mallows.center, 'phi_list': [mallows.phi]}, outfile, indent=4)

    # original cmd in terminal is java -Xmx500g -Xms4g -jar ltm.jar temp.json 48 >> out.txt 2>&1
    execute_jar = f'java -Xmx{calculate_jvm_xmx()}g -Xms4g -jar {jar_file} {json_file} {num_cores}'

    with open(ltm_verbose_file, 'a') as outfile:
        try:
            subprocess.run(execute_jar.split(), stdout=outfile, stderr=outfile, timeout=timeout)
            outfile.write('\n')
        except subprocess.TimeoutExpired:
            return False, 0, 0

    with open(json_file, 'r') as file:
        res = json.load(file)

    # # delete temp.json
    # subprocess.run(['rm', json_file])

    return True, res['prob_list'][0], res['runtime(ms)']


if __name__ == "__main__":
    from core.patterns import ItemPref

    model = Mallows(list(range(3)), 0.01)
    pref = ItemPref({2: {0}})
    res = calculate_marginal_prob_over_mallows_by_ltm(pref, model)
    print(res)

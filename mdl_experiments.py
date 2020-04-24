from printer import info
from plot_heatmap_of_differences import plot_heatmap
import os
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dfa import DFA
from dfa_annealer import DFA_Annealer
from binary_representation import get_binary_representation
from randomizer import Randomizer
from run_single_simulation import make_list_of_set_pairs_for_determiner_EXACTLY

ARBITRARY_REPETITIONS_ALL_OF_EXACTLY = 5


def get_positive_examples_for_exactly(
        randomizer,
        min_zeros_per_positive_example, max_zeros_per_positive_example,
        min_sample_for_each_n, max_sample_for_each_n, n1, n2):
    return [get_binary_representation(Randomizer(randomizer.seed), i, j).get_binary_representation(shuffle=True) for i, j in
            make_list_of_set_pairs_for_determiner_EXACTLY(
                (n1, n2),
                min_sample_for_each_n,
                max_sample_for_each_n,
                min_zeros_per_positive_example,
                max_zeros_per_positive_example)]


def create_dfa_all_of_the_exactly(n_values):
    states = ['qF'] + ['q' + str(i) for i in range(max(n_values) + 1)]
    transitions = defaultdict(dict)
    for i in range(max(n_values)):
        transitions['q' + str(i)]['1'] = 'q' + str(i + 1)
    for n in n_values:
        transitions['q' + str(n)]['#'] = 'qF'
    return DFA(
        states=states,
        transitions=transitions,
        initial='q0',
        accepting=['qF']
    )


# def create_dfa_all():
#     return DFA(
#         states = ['q0', 'qF'],
#         transitions={'q0': {'1': 'q0', '#': 'qF'}},
#         initial='q0',
#         accepting=['qF']
#     )


def create_dfa_init_hyp():
    return DFA(
        states = ['q0', 'qF'],
        transitions={'q0': {'0': 'q0', '1': 'q0', '#': 'qF'}},
        initial='q0',
        accepting=['qF']
    )


def create_dfa_exactly(n_values):
    states = ['qF'] + ['q' + str(i) for i in range(max(n_values) + 1)]
    transitions = defaultdict(dict)
    for i in range(max(n_values)):
        transitions['q' + str(i)]['0'] = 'q' + str(i)
        transitions['q' + str(i)]['1'] = 'q' + str(i + 1)
    transitions['q' + str(max(n_values))]['0'] = 'q' + str(max(n_values))
    for n in n_values:
        transitions['q' + str(n)]['#'] = 'qF'
    return DFA(
        states=states,
        transitions=transitions,
        initial='q0',
        accepting=['qF']
    )


# def compute_mdl_differences_all_vs_all_of_the_exactly(min_n, max_n):
#     results = {}
#     for n1 in range(min_n, max_n + 1):
#         for n2 in range(n1, max_n + 1):
#             results[n1, n2] = DFA_Annealer.compare_energy(
#                 create_dfa_all(),
#                 create_dfa_all_of_the_exactly((n1, n2)),
#                 sorted(['1' * n1, '1' * n2] * ARBITRARY_REPETITIONS_ALL_OF_EXACTLY)
#             )
#     return results

def compute_mdl_differences_init_hyp_vs_all_of_the_exactly(num_repetitions_of_each_positive_example, min_n, max_n):
    results = {}
    for n1 in range(min_n, max_n + 1):
        for n2 in range(n1, max_n + 1):
            results[n1, n2] = DFA_Annealer.energy_difference_a_minus_b(
                create_dfa_init_hyp(),
                create_dfa_all_of_the_exactly((n1, n2)),
                sorted(['1' * n1 + '#', '1' * n2 + '#'] * num_repetitions_of_each_positive_example)
            )
    return results


def compute_average_energy_difference_exactly_minus_init_hyp(
        randomizer,
        min_num_zeros_per_positive_example,
        max_num_zeros_per_positive_example,
        num_repeat, min_sample_for_each_n, max_sample_for_each_n, min_n, max_n):
    results = defaultdict(lambda: 0)
    for _ in range(num_repeat):
        for n1 in range(min_n, max_n + 1):
            for n2 in range(n1, max_n + 1):
                results[n1, n2] += DFA_Annealer.energy_difference_a_minus_b(
                    create_dfa_exactly((n1, n2)),
                    create_dfa_init_hyp(),
                    get_positive_examples_for_exactly(
                        randomizer,
                        min_num_zeros_per_positive_example,
                        max_num_zeros_per_positive_example,
                        min_sample_for_each_n, max_sample_for_each_n, n1, n2)
                )
    return {k: (v / num_repeat) for k, v in results.items()}


def plot_mdl_differences(gloal_min, global_max, title, image_file_name, max_n, matrix_as_dict):
    font = {'weight': 'bold', 'size': 15}
    matplotlib.rc('font', **font)
    mask = np.zeros((max_n + 1, max_n + 1))
    mask[:, 0] = True
    mask[0, :] = True
    matrix_as_array = np.zeros((max_n + 1, max_n + 1))
    for k, v in matrix_as_dict.items():
        matrix_as_array[k[1], k[0]] = v
        if k[0] != k[1]:
            mask[k[0], k[1]] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(20, 20))
        # cbar_ax = fig.add_axes([.905, .1, .05, .7])
        ax = sns.heatmap(matrix_as_array,
                         vmin=gloal_min, vmax=global_max,
                         ax=ax, mask=mask, square=True,
                         cmap='bwr_r',
                         # cbar_ax = cbar_ax, cbar=True
                         annot=True, fmt='.0f')
        ax.invert_yaxis()
        ax.set_xlim(xmin=1)
        ax.set_ylim(ymin=1)
        ax.set_title(title)
        ax.set_xlabel('$n_2$')
        ax.set_ylabel('$n_1$')
        plt.savefig(image_file_name)


def repeat_all_of_the_exactly(min_num_repeat_pos_ex, max_num_repeat_pos_ex, minimum_n, maximum_n):
    all_results = {num_repeat_pos_ex:
                       compute_mdl_differences_init_hyp_vs_all_of_the_exactly(num_repeat_pos_ex, minimum_n, maximum_n)
                   for num_repeat_pos_ex in range(min_num_repeat_pos_ex, max_num_repeat_pos_ex + 1)}
    for num_repeat_pos_ex in range(min_num_repeat_pos_ex, max_num_repeat_pos_ex + 1):
        plot_mdl_differences(
            min(map(lambda d: min(d.values()), all_results.values())),
            max(map(lambda d: max(d.values()), all_results.values())),
            'ALL_OF_THE_EXACTLY\n$E$(Initial DFA) - $E$(Target DFA)\n#Each Positive Example = %d' % num_repeat_pos_ex,
            'init_hyp_vs_all_of_the_exactly_min_%d_max_%d_rpt_%d.png' % (minimum_n, maximum_n, num_repeat_pos_ex),
            maximum_n,
            all_results[num_repeat_pos_ex]
        )


def plot_mdl_differences_for_determiner_exactly(
        randomizer,
        min_num_zeros_per_positive_example, max_num_zeros_per_positive_example,
        num_repeat, min_sample_for_each_n, max_sample_for_each_n, minimum_n, maximum_n):
    all_results = compute_average_energy_difference_exactly_minus_init_hyp(
        randomizer,
        min_num_zeros_per_positive_example, max_num_zeros_per_positive_example,
        num_repeat, min_sample_for_each_n, max_sample_for_each_n, minimum_n, maximum_n)
    plot_heatmap(
        False,
        minimum_n,
        maximum_n,
        '$E\\left(DFA^{EX}\\left(n_1, n_2\\right)\\right) - E\\left(DFA^{ALL}\\right)$' + '\nLower is Better',
        os.path.join('figures', 'average_energy_difference_exactly_%d_%d.png' %
                     (min_sample_for_each_n, max_sample_for_each_n)),
        maximum_n,
        all_results)


if __name__ == '__main__':
    # info('\n'.join(str(item) for item in sorted(compute_mdl_differences(1, 20).items(),
    #                                              key=lambda pair: pair[1])))
    import uuid
    randomizer = Randomizer(uuid.uuid1())
    min_n, max_n = 1, 20
    # repeat_all_of_the_exactly(1, 5, min_n, max_n)
    for number_of_examples_for_each_n in range(1, 15):
        plot_mdl_differences_for_determiner_exactly(
            randomizer,
            0, 100,
            10, number_of_examples_for_each_n, number_of_examples_for_each_n, min_n, max_n)

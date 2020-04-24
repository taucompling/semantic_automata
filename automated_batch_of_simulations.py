import os
import pickle
import uuid
from multiprocessing.pool import Pool
import pathlib
import numpy as np
from randomizer import Randomizer

from printer import info, set_up_logging
from run_single_simulation import SingleSimulationRunner, run_dir,get_binary_representation

_BASE_SEED = 100
_RANDOMIZER = Randomizer(_BASE_SEED)


def make_list_of_set_pairs_for_quantifier_between(at_least_ones, at_most_ones,
                                                  min_size_of_universe,
                                                  max_size_of_universe,
                                                  number_of_positive_examples,
                                                  add_examples_which_are_all_ones_of_these_lengths=[]):
    """
    Returns pairs, each of which will later be transformed into a binary string, which represents set membership.
    In each pair:
    1) First element is the universe: a set {0,...,M}, where M is a random number in [min_list_size, ..., max_list_size].
    2) Second element is a random subset {0,...,K}, where K is in [at_least, ..., at_most]. More precisely, K is in [at_least,..., M] if M < at_most.

    Thus generally speaking, the result is multiple instances of the generalized quantifier, each taken at random from a random size universe. Numeric example:
    at_least=3
    at_most=6
    min_list_size=20
    max_list_size=40
    number_of_lists=18
    We'll get 18 pairs. Example of one pair:
    (L1, L2) where
    L1 = {0, 1, 2, ..., 32}
    L2 = set(range(self.randomizer.get_prng().choice(range(3, min(7, 33))))) = set(range(self.randomizer.get_prng().choice(range(3, 7))) = set(range(22))

    :param at_least_ones:
    :param at_most_ones:
    :param min_size_of_universe:
    :param max_size_of_universe:
    :param number_of_positive_examples:
    :return:
    """
    if not all(at_least_ones <= length <= at_most_ones for length in add_examples_which_are_all_ones_of_these_lengths):
        raise ValueError('Length to add is out of allowed range')
    positive_examples_as_pairs_of_sets = []
    for i in range(number_of_positive_examples):
        universe_size = _RANDOMIZER.get_prng().choice(range(min_size_of_universe, max_size_of_universe))
        univese_set = set(range(universe_size))
        subset_of_universe = set(
            range(_RANDOMIZER.get_prng().choice(range(at_least_ones, min(at_most_ones + 1, universe_size)))))
        positive_examples_as_pairs_of_sets.append((univese_set, subset_of_universe))
    positive_examples_as_pairs_of_sets.extend(
        (set(range(length)), set(range(length))) for length in add_examples_which_are_all_ones_of_these_lengths)
    return positive_examples_as_pairs_of_sets


def make_list_of_set_pairs_for_quantifier_none(min_set_size, max_set_size, number_of_pairs):
    lists = []
    for i in range(number_of_pairs):
        list_size = _RANDOMIZER.get_prng().choice(range(min_set_size, max_set_size))
        lists.append((set(range(list_size)), set()))
    return lists


def run_single_simulation_for_multiprocessing(args_and_kwargs):
    args, kwargs = args_and_kwargs
    create_plots, seed, args_for_simulation = args[0], args[1], args[2:]
    set_up_logging('out.log')
    info('##### Task is:', args_and_kwargs)
    return SingleSimulationRunner(create_plots, seed).run_single_simulation(*args_for_simulation, **kwargs)


def run_batch(create_plots, base_seed,
              quantifier_type, initial_temperature, threshold, alpha,
              num_simulations,
              **kwargs):
    tasks = [((create_plots, seed, quantifier_type, initial_temperature, threshold, alpha), kwargs)
             for seed in range(base_seed, base_seed + num_simulations)]
    with Pool(maxtasksperchild=100) as pool:
        results = pool.map(run_single_simulation_for_multiprocessing, tasks)
        result_hypotheses, is_successes, output_dirs = zip(*results)
        total_success = sum(is_successes)
        info('########### Total success for quantifier %s is %d of %d' % (quantifier_type, total_success, num_simulations))
        with open(os.path.join(run_dir(quantifier_type, initial_temperature, alpha, threshold), 'num_success.csv'), 'w') as f_num_success:
            f_num_success.write('percent_success,total_success,num_simulations,quantifier,base_seed\n%f,%d,%d,%s,%s\n' %
                                (total_success / float(num_simulations), total_success, num_simulations, quantifier_type, base_seed))
        with open("result_hypotheses.pickle", "wb") as f:
            pickle.dump(result_hypotheses, f)

        mdl_scores = [x.get_mdl() for x in result_hypotheses]
        print(f"Mean MDL: {np.mean(mdl_scores)}, std {np.std(mdl_scores)}")
        best_hypothesis_idx = np.argmin(mdl_scores)
        best_hypothesis = result_hypotheses[best_hypothesis_idx]
        best_hypothesis_dir = str(pathlib.Path(output_dirs[best_hypothesis_idx]).parent)
        print(f"Best hypothesis dir: {best_hypothesis_dir}")
        best_hypothesis.plot_transitions('H_best', best_hypothesis_dir)

        return total_success


if __name__ == '__main__':
    # ============> FOR REPRODUCIBILITY, YOU MUST SET PYTHONHASHSEED=0 IN ENV BEFORE RUNNING <==========
    set_up_logging('out.log')
    also_plot = True
    info('-------------- STARTING BATCH OF SIMULATIONS WITH BASE SEED %s ---------' % _BASE_SEED)

    # run_batch(also_plot, base_seed, 'BETWEEN_WITH_DYNAMIC_UNIVERSE_SIZE', initial_temperature=100, threshold=1.0, alpha=0.99,
    #           num_simulations=10, min_set_size=5, max_set_size=30, number_of_pairs=30)

    min_set_size = 5
    max_set_size = 61
    number_of_pairs = 100
    data = make_list_of_set_pairs_for_quantifier_none(min_set_size=min_set_size,
                                                      max_set_size=max_set_size,
                                                      number_of_pairs=number_of_pairs)

    positive_examples = [get_binary_representation(_RANDOMIZER, set_a, set_b) for set_a, set_b in data]

    run_batch(True, _BASE_SEED, 'NONE',
              initial_temperature=100,
              threshold=1.0,
              alpha=0.99,
              num_simulations=64,
              min_set_size=min_set_size,
              max_set_size=max_set_size,
              positive_examples=positive_examples,
              number_of_pairs=number_of_pairs)

    # run_batch(also_plot, base_seed, 'BETWEEN_WITH_FIXED_UNIVERSE_SIZE', 2000, 1.0, 0.95,
    #      num_simulations=10,
    #      all_ones=[],
    #      at_least_ones=3, at_most_plus_1_ones=6, fixed_universe_size=10,
    #      number_of_positive_examples=50)
    #

    alpha = 0.99
    num_simulations = 1,
    add_examples_which_are_all_ones_of_these_lengths = []
    at_least_ones = 3
    at_most_ones = 6
    min_size_of_universe = 10
    max_size_of_universe = 30
    number_of_positive_examples = 100



    # data = make_list_of_set_pairs_for_quantifier_between(at_least_ones=at_least_ones, at_most_ones=at_most_ones,
    #                                                      min_size_of_universe=min_size_of_universe, max_size_of_universe=max_size_of_universe,
    #                                                      number_of_positive_examples=number_of_positive_examples,
    #                                                      add_examples_which_are_all_ones_of_these_lengths=add_examples_which_are_all_ones_of_these_lengths)
    #
    # positive_examples = [get_binary_representation(_RANDOMIZER, set_a, set_b) for set_a, set_b in data]
    # run_batch(also_plot, _BASE_SEED, 'BETWEEN_WITH_DYNAMIC_UNIVERSE_SIZE',
    #           initial_temperature=100,
    #           threshold=1.0,
    #           alpha=0.99,
    #           num_simulations=64,
    #           positive_examples=positive_examples,
    #           at_least_ones=at_least_ones, at_most_ones=at_most_ones,
    #           min_size_of_universe=min_size_of_universe, max_size_of_universe=max_size_of_universe,
    #           number_of_positive_examples=number_of_positive_examples,
    #           add_examples_which_are_all_ones_of_these_lengths=add_examples_which_are_all_ones_of_these_lengths)
    # #
    # run_batch(also_plot, base_seed, 'EXACTLY', 3700, 1.0, 0.96,
    #      num_simulations=10,
    #      ns=(2, 5, 9),
    #      min_sample_for_each_n=5,
    #      max_sample_for_each_n=10,
    #      min_zeros_per_positive_example=0,
    #      max_zeros_per_positive_example=20)
    #
    # run_batch(also_plot, base_seed, 'ALL_OF_THE_EXACTLY', 3800, 1.0, 0.97,
    #      num_simulations=10,
    #      ns=(2, 5, 9), min_sample_for_each_n=5, max_sample_for_each_n=10)

    info('-------------- FINISHED BATCH OF SIMULATIONS WITH BASE SEED %s ---------' % _BASE_SEED)

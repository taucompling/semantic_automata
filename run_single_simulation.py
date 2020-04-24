from printer import info
import uuid
from randomizer import Randomizer
import glob
import os
import pathlib
import shutil
from dfa_annealer import DFA_Annealer
from simulated_annealing import Simulated_annealing_learner
from binary_representation import get_binary_representation
import target_automaton


def run_dir(quantifier_type, initial_temperature, alpha, threshold):
    return os.path.join('sa_simulations',
                        quantifier_type,
                        ('tempinit[%s]alpha[%s]thresh[%s]' % (initial_temperature, alpha, threshold)))


class SingleSimulationRunner(object):
    def __init__(self, create_plots, seed):
        self.create_plots = create_plots
        self.randomizer = Randomizer(seed)

    def make_list_of_set_pairs_for_determiner_EXACTLY(self,
            ns, min_sample_for_each_n, max_sample_for_each_n,
            min_zeros_per_positive_example, max_zeros_per_positive_example):
        pairs = []
        for n in ns:
            pairs.extend([(set(range(n + self.randomizer.get_prng().randint(min_zeros_per_positive_example, max_zeros_per_positive_example))),
                           set(range(n)))
                          for _ in range(self.randomizer.get_prng().randint(min_sample_for_each_n, max_sample_for_each_n))
                          ])
        return pairs


    def simulate_EXACTLY(self, initial_temperature, threshold, alpha,
                         ns, min_sample_for_each_n, max_sample_for_each_n,
                         min_zeros_per_positive_example, max_zeros_per_positive_example):
        data = self.make_list_of_set_pairs_for_determiner_EXACTLY(
                ns, min_sample_for_each_n, max_sample_for_each_n,
                min_zeros_per_positive_example, max_zeros_per_positive_example)
        return self.__simulate_with_data('EXACTLY',
                                    dict(
                                            ns=ns,
                                            min_sample_for_each_n=min_sample_for_each_n,
                                            max_sample_for_each_n=max_sample_for_each_n,
                                            min_zeros_per_positive_example=min_zeros_per_positive_example,
                                            max_zeros_per_positive_example=max_zeros_per_positive_example),
                                    data, initial_temperature, threshold, alpha)


    def make_list_of_set_pairs_quantifier_ALL_OF_THE_EXACTLY(self, ns, min_sample_for_each_n, max_sample_for_each_n):
        pairs = []
        for n in ns:
            pairs.extend([(set(range(n)), set(range(n))) for _ in range(
                    self.randomizer.get_prng().randint(min_sample_for_each_n, max_sample_for_each_n))])
        return pairs


    def simulate_ALL_OF_THE_EXACTLY(self, initial_temperature, threshold, alpha,
                                    ns, min_sample_for_each_n, max_sample_for_each_n):
        data = self.make_list_of_set_pairs_quantifier_ALL_OF_THE_EXACTLY(ns, min_sample_for_each_n, max_sample_for_each_n)
        return self.__simulate_with_data(
                'ALL_OF_THE_EXACTLY',
                dict(ns=ns, min_sample_for_each_n=min_sample_for_each_n, max_sample_for_each_n=max_sample_for_each_n),
                data, initial_temperature, threshold, alpha)


    def make_list_of_set_pairs_for_quantifier_all(self, min_set_size, max_set_size, number_of_pairs):
        lists = []
        for i in range(number_of_pairs):
            list_size = self.randomizer.get_prng().choice(range(min_set_size, max_set_size))
            lists.append((set(range(list_size)), set(range(list_size))))
        return lists


    def make_list_of_set_pairs_for_quantifier_none(self, min_set_size, max_set_size, number_of_pairs):
        lists = []
        for i in range(number_of_pairs):
            list_size = self.randomizer.get_prng().choice(range(min_set_size, max_set_size))
            lists.append((set(range(list_size)), set()))
        return lists


    def make_list_of_set_pairs_for_quantifier_between(self, at_least_ones, at_most_ones,
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
            universe_size = self.randomizer.get_prng().choice(range(min_size_of_universe, max_size_of_universe))
            univese_set = set(range(universe_size))
            subset_of_universe = set(range(self.randomizer.get_prng().choice(range(at_least_ones, min(at_most_ones + 1, universe_size)))))
            positive_examples_as_pairs_of_sets.append((univese_set, subset_of_universe))
        positive_examples_as_pairs_of_sets.extend(
                (set(range(length)), set(range(length))) for length in add_examples_which_are_all_ones_of_these_lengths)
        return positive_examples_as_pairs_of_sets


    def simulate_whatever(self):
        DOGS = {'Rex', 'Spot', 'Bolt', 'Belka', 'Laika', 'Azit'}
        BROWN_ANIMALS = {'Belka', 'Spot', 'Azit', 'Mitzi'}
        SATELLITES = {'Yaogan', 'Ofeq_7', 'Ofeq_9', 'WorldView', 'Eros_B', 'Amos_5', 'Glonass'}
        LOW_EARTH_ORBIT = {'Yaogan', 'Ofeq_7', 'Ofeq_9', 'WorldView', 'Eros_B', 'Hubble'}
        data = [(DOGS, BROWN_ANIMALS), (SATELLITES, LOW_EARTH_ORBIT)]

        DOGS = {'Rex', 'Spot', 'Bolt', 'Belka', 'Laika', 'Azit'}
        BROWN_ANIMALS = {'Rex', 'Spot', 'Bolt', 'Belka', 'Laika', 'Azit', 'IKEA table', 'Humus'}
        SATELLITES = {'Yaogan', 'Ofeq_7', 'Ofeq_9', 'WorldView', 'Eros_B'}
        LOW_EARTH_ORBIT = {'Yaogan', 'Ofeq_7', 'Ofeq_9', 'WorldView', 'Eros_B', 'Hubble'}
        BOYS = {'Tom', 'John', 'Max', 'Mark', 'Barak', 'Guy', 'Ted', 'Joey'}
        HAPPY = {'Linda', 'Mary', 'Tom', 'John', 'Max', 'Mark', 'Barak', 'Guy', 'Ted', 'Joey'}
        data5 = [(DOGS, BROWN_ANIMALS), (SATELLITES, LOW_EARTH_ORBIT), (BOYS, HAPPY)]  # ALL

        GROUP_A = {'hello'}
        GROUP_B = {'hello'}
        GROUP_C = {'0', '2', '6', '17'}
        GROUP_D = {'0', '2', '6', '17'}
        data_1 = [(GROUP_A, GROUP_B), (GROUP_C, GROUP_D)]  # Minimal Quantifier: NONE

        CATS = {'Mitzi', 'Tuli', 'KitKat', 'Chat', 'Ears'}
        TWEET = {'Tweety', 'Zebra Finch', 'Cockatoo'}
        HAVE_ONE_SOUL = {'Rex', 'John'}
        data7 = [(CATS, TWEET), (CATS, HAVE_ONE_SOUL), (CATS, DOGS)]


    def simulate_data_3(self):
        def make_list_of_set_pairs_2(at_least_not, at_most_not, list_size):
            Q = frozenset(range(1, at_most_not * 2))
            return [(frozenset(self.randomizer.get_prng().sample(Q, len(Q) - self.randomizer.get_prng().randint(at_least_not, at_most_not))), Q) for i in
                    range(list_size)]

        data3 = make_list_of_set_pairs_2(at_least_not=3, at_most_not=7, list_size=50)
        assert all(len(Q) - 10 <= len(P) <= len(Q) - 0 for P, Q in data3)


    def create_output_directory(self, quantifier_type, additional_parameters_to_persist,
                                positive_examples, initial_temperature, threshold, alpha):
        output_directory = os.path.join(run_dir(quantifier_type, initial_temperature, alpha, threshold),
                                        ('runid[%s]' % uuid.uuid4().hex))
        os.makedirs(output_directory)
        with open(os.path.join(output_directory, 'parameters.csv'), 'w') as params_f:
            params_f.write('initial_temperature,%s\n' % initial_temperature)
            params_f.write('threshold,%s\n' % threshold)
            params_f.write('alpha,%s\n' % alpha)
            for param_name, param_value in additional_parameters_to_persist.items():
                params_f.write('%s,%s\n' % (param_name, param_value))
        with open(os.path.join(output_directory, 'positive_examples.txt'), 'w') as pos_f:
            pos_f.write('\n'.join(positive_examples))
        return output_directory


    def __simulate_with_data(self, quantifier_type, additional_parameters_to_persist,
                             positive_examples, initial_temperature, threshold, alpha):
        output_directory = self.create_output_directory(quantifier_type, additional_parameters_to_persist,
                                                   positive_examples, initial_temperature, threshold, alpha)
        annealer = DFA_Annealer(self.randomizer.seed)
        learner = Simulated_annealing_learner(self.randomizer.seed, initial_temperature, positive_examples, annealer)
        final_hyp = learner.logger(self.create_plots, positive_examples, output_directory, threshold, alpha)[0]
        final_hyp.positive_examples = positive_examples
        return output_directory, final_hyp, positive_examples


    def simulate_BETWEEN_with_dynamic_universe_size(self, initial_temperature, threshold, alpha,
                                                    add_examples_which_are_all_ones_of_these_lengths,
                                                    at_least_ones, at_most_ones,
                                                    min_size_of_universe, max_size_of_universe,
                                                    number_of_positive_examples, positive_examples):
        return self.__simulate_with_data(
                'BETWEEN_WITH_DYNAMIC_UNIVERSE_SIZE',
                dict(add_examples_which_are_all_ones_of_these_lengths=add_examples_which_are_all_ones_of_these_lengths,
                     at_least_ones=at_least_ones, at_most_ones=at_most_ones,
                     min_size_of_universe=min_size_of_universe, max_size_of_universe=max_size_of_universe,
                     number_of_positive_examples=number_of_positive_examples),
                positive_examples, initial_temperature, threshold, alpha)


    def simulate_BETWEEN_with_fixed_universe_size(self, initial_temperature, threshold, alpha, all_ones,
                                                  at_least_ones, at_most_plus_1_ones,
                                                  fixed_universe_size, number_of_positive_examples):
        data = self.make_list_of_set_pairs_for_quantifier_between(at_least_ones, at_most_plus_1_ones,
                                                             min_size_of_universe=fixed_universe_size,
                                                             max_size_of_universe=fixed_universe_size + 1,
                                                             number_of_positive_examples=number_of_positive_examples,
                                                             add_examples_which_are_all_ones_of_these_lengths=all_ones)
        return self.__simulate_with_data(
                'BETWEEN_WITH_FIXED_UNIVERSE_SIZE',
                dict(all_ones=all_ones,
                     at_least_ones=at_least_ones, at_most_plus_1_ones=at_most_plus_1_ones,
                     fixed_universe_size=fixed_universe_size, number_of_positive_examples=number_of_positive_examples),
                data, initial_temperature, threshold, alpha)


    def simulate_ALL(self, initial_temperature, threshold, alpha,
                     min_set_size, max_set_size, number_of_pairs):
        data = self.make_list_of_set_pairs_for_quantifier_all(min_set_size, max_set_size, number_of_pairs)
        return self.__simulate_with_data(
                'ALL',
                dict(min_set_size=min_set_size, max_set_size=max_set_size, number_of_pairs=number_of_pairs),
                data, initial_temperature, threshold, alpha)


    def simulate_NONE(self, initial_temperature, threshold, alpha,
                      min_set_size, max_set_size, number_of_pairs, positive_examples):
        return self.__simulate_with_data(
                'NONE',
                dict(min_set_size=min_set_size, max_set_size=max_set_size, number_of_pairs=number_of_pairs),
                positive_examples, initial_temperature, threshold, alpha)

    def run_single_simulation(self, quantifier_type,
                              initial_temperature,
                              threshold,
                              alpha,
                              *args, **kwargs):
        info('############ Starting simulation for quantifier %s' % quantifier_type)
        quantifier_names_to_functions = {
            'BETWEEN_WITH_DYNAMIC_UNIVERSE_SIZE': self.simulate_BETWEEN_with_dynamic_universe_size,
            'NONE': self.simulate_NONE,
            'EXACTLY': self.simulate_EXACTLY,
            'ALL': self.simulate_ALL,
            'ALL_OF_THE_EXACTLY': self.simulate_ALL_OF_THE_EXACTLY,
            'BETWEEN_WITH_FIXED_UNIVERSE_SIZE': self.simulate_BETWEEN_with_fixed_universe_size
        }
        qunatifier_names_to_target_dfa = {
            'NONE': target_automaton.expected_final_hyp_none(),
            'ALL': target_automaton.expected_final_hyp_all(),
            'BETWEEN_WITH_FIXED_UNIVERSE_SIZE':
                target_automaton.expected_final_hyp_between_with_any_universe_size(
                    lower=kwargs.get('at_least_ones'),
                    upper=kwargs.get('at_most_plus_1_ones')),
            'BETWEEN_WITH_DYNAMIC_UNIVERSE_SIZE':
                target_automaton.expected_final_hyp_between_with_any_universe_size(
                    lower=kwargs.get('at_least_ones'),
                    upper=kwargs.get('at_most_ones')),
            'EXACTLY': target_automaton.expected_final_hyp_exactly(kwargs.get('ns')),
            'ALL_OF_THE_EXACTLY': target_automaton.expected_final_hyp_all_of_the_exactly(kwargs.get('ns'))
        }
        if quantifier_type in quantifier_names_to_functions:
            output_directory, final_hyp, positive_examples = quantifier_names_to_functions[quantifier_type] \
                (initial_temperature, threshold, alpha, *args, **kwargs)
            final_hyp.plot_transitions('H_final', output_directory)
            is_success = (final_hyp == qunatifier_names_to_target_dfa[quantifier_type])
            with open(os.path.join(output_directory, 'is_success.txt'), 'w') as f_success:
                f_success.write(str(is_success))
            # with open(os.path.join(output_directory, 'energy_final_hyp_minus_target.csv'), 'w') as final_diff_f:
            #     final_diff_f.write(str(DFA_Annealer.energy_difference_a_minus_b(
            #             final_hyp,
            #             qunatifier_names_to_target_dfa[quantifier_type],
            #             positive_examples)) if quantifier_type in qunatifier_names_to_target_dfa \
            #                            else 'No target automaton defined')
            info('############ Finished simulation for quantifier %s, output in %s' % (quantifier_type, output_directory))
            qunatifier_names_to_target_dfa[quantifier_type].positive_examples = positive_examples
            qunatifier_names_to_target_dfa[quantifier_type].plot_transitions("H_target", str(pathlib.Path(output_directory).parent))

            print(f"Target hypothesis MDL: {qunatifier_names_to_target_dfa[quantifier_type].get_mdl()}")
            return final_hyp, is_success, output_directory
        else:
            raise ValueError('Unknown quantifier type %s' % quantifier_type)


if __name__ == "__main__":
    #    shutil.rmtree('./figures')
    ##
    ##    info("# APPLYING LEARNER ON THE FOLLOWING PAIRS OF SETS: ")
    ##    pair_counter = 1
    ##    for set_tuple in data:
    ##        info("Pair no.", pair_counter)
    ##        info(set_tuple)
    ##        R = Relation(set_tuple[0], set_tuple[1])
    ##        info("Binary representation of pair:", R.get_bianry_representation())
    ##        pair_counter += 1

    initial_temperature = 2000
    threshold = 1.0
    alpha = 0.95
    number_of_pairs = 50
    # simulate_between_3_and_6(initial_temperature=2000, threshold=1.0, alpha=0.95, all_ones=[4])

    # simulate_all(initial_temperature=2000, threshold=1.0, alpha=0.95,
    #   max_set_size=61, number_of_pairs=50)

    # simulate_none(initial_temperature=2000, threshold=1.0, alpha=0.95,
    #               min_set_size=5, max_set_size=61, number_of_pairs=50)

    # simulate_BETWEEN_with_fixed_universe_size(initial_temperature, threshold, alpha,
    #                                           all_ones=[],
    #                                           at_least_ones=3, at_most_plus_1_ones=6, fixed_universe_size=10,
    #                                           number_of_positive_examples=number_of_pairs)

    # simulate_ALL_OF_THE_EXACTLY(initial_temperature, threshold, alpha,
    #                             ns=(2, 5, 9), min_sample_for_each_n=5, max_sample_for_each_n=10)

    # SingleSimulationRunner(0).run_single_simulation('EXACTLY', initial_temperature, threshold, alpha,
    #                       ns=(2, 5, 9), min_sample_for_each_n=5, max_sample_for_each_n=10,
    #                       min_zeros_per_positive_example=0, max_zeros_per_positive_example=20)
    #
    # SingleSimulationRunner(0).run_single_simulation('ALL', initial_temperature=2000, threshold=1.0, alpha=0.95,
    #                       min_set_size=5, max_set_size=61, number_of_pairs=50)
    #
    SingleSimulationRunner(create_plots=False, seed=0).run_single_simulation('BETWEEN_WITH_DYNAMIC_UNIVERSE_SIZE',
                                                                             initial_temperature=2,
                                                                             threshold=1.0,
                                                                             alpha=0.99,
                                                                             add_examples_which_are_all_ones_of_these_lengths=[],
                                                                             at_least_ones=3, at_most_ones=6,
                                                                             min_size_of_universe=10,
                                                                             max_size_of_universe=30,
                                                                             number_of_positive_examples=100)
    #
    # simulate_BETWEEN_with_dynamic_universe_size(initial_temperature, threshold, alpha,
    #                                             add_examples_which_are_all_ones_of_these_lengths=[],
    #                                             at_least_ones=5, at_most_ones=61, min_size_of_universe=20,
    #                                             max_size_of_universe=80, number_of_positive_examples=50)

from randomizer import Randomizer
from printer import info
import datetime
import math


class Simulated_annealing_learner:
    def __init__(self, seed, initial_t, positive_examples, annealer):
        self.randomizer = Randomizer(seed)
        self.annealer = annealer
        self.T = initial_t
        self.positive_examples = positive_examples
        # self.hyp = self.annealer.find_initial_hypthesis(data)
        self.hyp = self.annealer.initial_hypothesis()
        self.creation_time = datetime.datetime.now()

    def simulated_annealing(self, create_plots, positive_examples, output_directory, threshold, alpha):
        assert (0 < alpha < 1) and (0 < threshold) and (self.T > 0)
        iter_counter = 0
        p = None

        self.hyp.positive_examples = positive_examples

        if create_plots:
            # Initial hypothesis
            self.hyp.plot_transitions(f'H{iter_counter}', output_directory)

        while self.T > threshold:
            iter_counter += 1 
            info("# ITERATION COUNTER =" , iter_counter)
            info("Current temperature:", self.T)
            H_tag = self.annealer.get_random_neighbor(self.hyp, self.positive_examples)
            delta = self.annealer.energy_difference_a_minus_b(H_tag, self.hyp, positive_examples)
            info("Delta =", delta)
            if delta < 0:
                p = 1
            else:
                p = math.exp(-delta/self.T)
            if p >= self.randomizer.get_prng().random():
                info("Changing hypothesis\n")
                self.hyp = H_tag
            else:
                info("Didn't change hypothesis\n")
            if create_plots:
                self.hyp.plot_transitions(f'H{iter_counter}', output_directory)
            self.T *= alpha
        info("CHOSEN HYPOTHESIS:\n", self.hyp)

        return self.hyp, positive_examples

    def logger(self, create_plots, positive_examples, output_directory, threshold, alpha):
        info("# APPLYING LEARNER ON THE FOLLOWING POSITIVE EXAMPLES: %s" % ','.join(positive_examples))
        info("\nInitial temperature:", self.T, ", Threshold:", threshold, ", Alpha:" , alpha)
        info("\n# INITIAL HYPOTHESIS:\n", self.hyp)
        info("\n")
        return self.simulated_annealing(create_plots, positive_examples, output_directory, threshold, alpha)

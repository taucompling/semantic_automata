from random import Random


class Randomizer(object):
    def __init__(self, seed):
        self.seed = seed
        self.prng = Random(seed)

    def get_prng(self):
        return self.prng

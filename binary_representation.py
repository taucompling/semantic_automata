def get_binary_representation(shuffler, set_a, set_b):
    """
    Returns a binary string that represents whether each member of the first set is in
    the intersection of the first set and the secound set

    @params shuffler: randomizer if ones and zeros need to be shuffled in the binary representation, otherwise the order of ones and zeros is arbitrary.
    """
    binary_representation = ["1" if member in (set_a & set_b) else "0" for member in set_a]
    if shuffler:
        shuffler.get_prng().shuffle(binary_representation)
    return ''.join(binary_representation) + '#'

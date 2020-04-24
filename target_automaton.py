from dfa import DFA


def expected_final_hyp_none():
    return DFA(
        states={'q0', 'qF'},
        transitions={'q0': {'0': 'q0', '#': 'qF'}},
        initial='q0',
        accepting={'qF'}
    )


def expected_final_hyp_all():
    return DFA(
        states={'q0', 'qF'},
        transitions={'q0': {'1': 'q0', '#': 'qF'}},
        initial='q0',
        accepting={'qF'}
    )


def expected_final_hyp_between_with_any_universe_size(lower, upper):
    def transition_before_lower(i):
        return ('q%d' % i), {'0': ('q%d' % i), '1': ('q%d' % (i + 1))}

    def transition_before_upper(i):
        return ('q%d' % i), {'0': ('q%d' % i), '1': ('q%d' % (i + 1)), '#': 'qF'}

    def transition_upper():
        return ('q%d' % upper), {'0': ('q%d' % upper), '#': 'qF'}

    return None if None in (lower, upper) else DFA(
        states={'q%d' % i for i in range(upper + 1)} | {'qF'},
        transitions=dict(
            [transition_before_lower(i) for i in range(lower)] +
            [transition_before_upper(i) for i in range(lower, upper)] +
            [transition_upper()]),
        initial='q0',
        accepting={'qF'}
    )


def expected_final_hyp_exactly(ns):
    def transition_not_in_ns(i):
        return ('q%d' % i), {'0': ('q%d' % i), '1': ('q%d' % (i + 1))}

    def transition_in_ns_except_max_n(i):
        return ('q%d' % i), {'0': ('q%d' % i), '1': ('q%d' % (i + 1)), '#': 'qF'}

    def transition_max_n():
        return ('q%d' % max(ns)), {'0': ('q%d' % max(ns)), '#': 'qF'}

    return None if ns is None else DFA(
        states={'q%d' % i for i in range(max(ns) + 1)} | {'qF'},
        transitions=dict(
            [transition_not_in_ns(i) for i in range(max(ns) + 1) if i not in ns] +
            [transition_in_ns_except_max_n(i) for i in ns if i != max(ns)] +
            [transition_max_n()]),
        initial='q0',
        accepting={'qF'}
    )


def expected_final_hyp_all_of_the_exactly(ns):
    def transition_not_in_ns(i):
        return ('q%d' % i), {'1': ('q%d' % (i + 1))}

    def transition_in_ns_except_max_n(i):
        return ('q%d' % i), {'1': ('q%d' % (i + 1)), '#': 'qF'}

    def transition_max_n():
        return ('q%d' % max(ns)), {'#': 'qF'}

    return None if ns is None else DFA(
        states={'q%d' % i for i in range(max(ns) + 1)} | {'qF'},
        transitions=dict(
            [transition_not_in_ns(i) for i in range(max(ns) + 1) if i not in ns] +
            [transition_in_ns_except_max_n(i) for i in ns if i != max(ns)] +
            [transition_max_n()]),
        initial='q0',
        accepting={'qF'}
    )


if __name__ == '__main__':
    # expected_final_hyp_between_with_any_universe_size(3, 6)\
    #     .plot_transitions('expected_final_hyp_between_3_and_6.png', '.')
    # expected_final_hyp_exactly((2, 5, 9))\
    #     .plot_transitions('expected_final_hyp_exactly.png', '.')
    expected_final_hyp_all_of_the_exactly((2, 5, 9))\
        .plot_transitions('expected_final_hyp_all_of_the_exactly.png', '.')

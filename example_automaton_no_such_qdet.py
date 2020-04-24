from dfa import DFA


dfa_no_such_generalized_quantificational_determiner = DFA(
            states={'q0', 'q1', 'q2', 'qF'},
            transitions={
                'q0': {'0': 'q1', '1': 'q2'},
                'q1': {'0': 'q1', '1': 'q1'},
                'q2': {'0': 'q2', '1': 'q2', '#': 'qF'},
            },
            initial='q0',
            accepting='qF'
        )
dfa_no_such_generalized_quantificational_determiner.plot_transitions(
    'No such Q-Det', '.')

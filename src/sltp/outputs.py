import logging


def print_transition_matrix(sample, transitions_filename):
    state_ids = list(range(0, len(sample.states)))
    transitions = sample.transitions
    logging.info(f"Printing transition data with {len(state_ids)} states"
                 f" and {len(transitions)} transitions to '{transitions_filename}'")

    with open(transitions_filename, 'w') as f:
        # first line: <#states> <#transitions>
        print(f"{len(state_ids)} {len(transitions)}", file=f)

        # Next lines: See format description in C++ code, at TransitionSample::read()
        for (sid, succ, label) in sample.transitions:
            print(f"{sid} {succ} {int(label)}", file=f)

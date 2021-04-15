import logging


def print_transition_matrix(sample, transitions_filename):
    state_ids = sample.get_sorted_state_ids()
    transitions = sample.transitions
    num_transitions = sum(len(targets) for targets in transitions.values())
    num_expanded = sum(1 for s in state_ids if sample.is_expanded(s))
    logging.info(f"Printing transition data with {len(state_ids)} states,"
                 f" {num_expanded} expanded states and {num_transitions} transitions to '{transitions_filename}'")

    # Make sure that state IDs start at 0 and are contiguous
    assert state_ids == list(range(0, len(state_ids)))

    with open(transitions_filename, 'w') as f:
        # first line: <#states> <#transitions>
        print(f"{len(state_ids)} {num_transitions}", file=f)

        # Next lines: See format description in C++ code, at TransitionSample::read()
        for s in state_ids:
            successors = transitions[s]
            sorted_succs = " ".join(map(str, sorted(successors)))
            line = f"{s}"
            line += f" {int(sample.is_expanded(s))}"
            line += f" {int(sample.is_goal(s))}"
            line += f" {int(sample.is_unsolvable(s))}"
            line += f" {sample.vstar.get(s, -1)}"
            line += f" {len(successors)}"
            line += f" {sorted_succs}"
            print(line, file=f)

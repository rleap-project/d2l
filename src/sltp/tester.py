import logging
import sys

from .models import FeatureModel
from .separation import TransitionClassificationPolicy, generate_user_provided_policy
from .features import generate_model_cache, create_model_factory, compute_static_atoms
from .returncodes import ExitCode
from .sampling import read_transitions_from_files


def apply_policy_on_test_instances(config, create_policy):
    """ Run a search on the test instances that follows the given policy """
    logging.info(f"Testing learnt policy on instances: {config.test_policy_instances}")
    pyperplan = _import_pyperplan(config.pyperplan_path)

    for instance in config.test_policy_instances:
        logging.info(f'Testing policy on instance "{instance}"')

        try:
            _run_pyperplan(pyperplan, config.test_domain, instance, create_policy, config.parameter_generator)
        except PolicySearchException as e:
            logging.warning(f"Testing of policy failed with code: {e.code}")
            return e.code
    logging.info("Learnt policy solves all test instances")
    return ExitCode.Success


def _import_pyperplan(path):
    sys.path.insert(0, path)
    import pyperplan
    sys.path = sys.path[1:]
    return pyperplan


def _run_pyperplan(pyperplan, domain, instance, create_policy, parameter_generator):
    # Let's fake a pyperplan command-line arguments object so that
    # we don't have to worry about default parameters. We use "gbf" but will override that next.
    args = pyperplan.parse_args([domain, instance])

    # Parse the domain & instance and create a model generator
    problem, model_factory = create_model_factory(domain, instance, parameter_generator)
    static_atoms, _ = compute_static_atoms(problem)

    # Compute an actual policy object that returns the next action to be applied
    search_policy = create_policy(model_factory, static_atoms)

    # And now we inject our desired search and heuristic functions
    args.forced_search = create_pyperplan_policy_based_search(pyperplan, search_policy)

    # And run pyperplan!
    pyperplan.main(args)


def translate_atom(atom):
    assert atom[0] == '(' and atom[-1] == ')'
    return atom[1:-1].split(' ')


def translate_state(state, static_atoms):
    """ Translate a pyperplan-like state into a list with the format required by SLTP's concept denotation processor """
    return [translate_atom(atom) for atom in state] + list(static_atoms)


def generate_model_from_state(model_factory, state, static_atoms):
    translated = translate_state(state, static_atoms)
    return FeatureModel(model_factory.create_model(translated))


def create_pyperplan_policy_based_search(pyperplan, search_policy):
    """ Creates a pyperplan like search object that uses the abstract generalized policy to perform the search """
    searchspace = pyperplan.search.searchspace

    def policy_based_search(task):  # The actual search object
        expanded = 0  # Some bookkeping
        node = searchspace.make_root_node(task.initial_state)
        closed = set()

        while not task.goal_reached(node.state):
            # logging.info(f'Expanding: {sorted(node.state)}')

            closed.add(node.state)
            expanded += 1
            if expanded % 1000 == 0:
                logging.debug(f"Number of expanded states so far in policy-based search: {expanded}")

            successors = [(op, op.apply(node.state)) for op in task.operators if op.applicable(node.state)]
            op, succ = search_policy(node.state, successors)

            if succ in closed:  # loop detection
                logging.error(f"Policy incurred in a loop after {expanded} expansions. Repeated node: {succ}")
                logging.error(f"Trajectory from initial state: "
                              f"{searchspace.make_child_node(node, op, succ).extract_solution()}")
                raise PolicySearchException(ExitCode.AbstractPolicyNonTerminatingOnTestInstances)

            node = searchspace.make_child_node(node, op, succ)

        logging.info(f"Goal found after expanding {expanded} nodes")
        return node.extract_solution()

    return policy_based_search


class PolicySearchException(Exception):
    def __init__(self, code):
        super().__init__("")
        self.code = code


def create_action_selection_function_from_transition_policy(model_factory, static_atoms, policy):
    assert isinstance(policy, TransitionClassificationPolicy)

    def _policy(state, successors):
        m0 = generate_model_from_state(model_factory, state, static_atoms)

        for op, succ in successors:
            m1 = generate_model_from_state(model_factory, succ, static_atoms)
            if policy.transition_is_good(m0, m1):
                return op, succ

        # No transition labeled as good by the policy
        logging.warning(f"Policy is incomplete on state:\n{state}")

        # Report the reason why no transition is labeled as good
        for op, succ in successors:
            m1 = generate_model_from_state(model_factory, succ, static_atoms)
            print(f"\nTransition to state {succ} is bad because:")
            policy.explain_why_transition_is_bad(m0, m1)

        raise PolicySearchException(ExitCode.AbstractPolicyNotCompleteOnTestInstances)

    return _policy


def test_d2l_policy(config, data, rng):
    if config.test_domain is None:
        logging.info("No testing instances were specified")
        return ExitCode.Success, dict()

    def create_policy(model_factory, static_atoms):
        if config.d2l_policy is not None:
            policy = generate_user_provided_policy(config)
        else:
            policy = data.d2l_policy

        return create_action_selection_function_from_transition_policy(model_factory, static_atoms, policy)

    # Test that the policy reaches a goal when applied on all test instances
    res = apply_policy_on_test_instances(config, create_policy)
    if res != ExitCode.Success:
        return res, dict()

    res = test_d2l_policy_is_complete(config, data.d2l_policy)
    if res != ExitCode.Success:
        return res, dict()
    logging.info("The computed policy solves all test instances and is complete in all sampled test states")
    return res, dict()


def test_d2l_policy_is_complete(config, policy):
    if not config.test_sample_files:
        logging.info("No test instances specified for testing of policy completeness")
        return ExitCode.Success

    logging.info(f"Testing learnt policy on sample of states from instances: {config.test_instances}")
    sample, _ = read_transitions_from_files(config.test_sample_files)
    _, model_cache = generate_model_cache(config.test_domain, config.test_instances, sample, config.parameter_generator)

    for s in sample.expanded:
        is_goal = s in sample.goals
        is_alive = s in sample.alive_states

        if not is_alive:
            # ATM we only check that the policy is complete on alive states and that it does not mandate an action
            # leading into a non-alive state.
            continue

        m0 = model_cache.get_feature_model(s)

        good_action_found = False
        for t in sample.transitions[s]:
            m1 = model_cache.get_feature_model(t)
            if policy.transition_is_good(m0, m1):
                if is_goal:
                    logging.error(f"Policy advises transition ({s}, {t}), but {s} is a goal state!"
                                  f"\ns: {sample.states[s]}\nt: {sample.states[t]}")
                    return ExitCode.SeparationPolicyCannotDistinguishGoal

                if t not in sample.alive_states:
                    logging.error(f"Policy advises transition ({s}, {t}), but {t} is dead!"
                                  f"\ns: {sample.states[s]}\nt: {sample.states[t]}")
                    return ExitCode.SeparationPolicyAdvisesDeadState

                good_action_found = True
                break

        if not is_goal and not good_action_found and len(sample.transitions[s]) > 0:
            logging.error(f"Policy incomplete on test state {s}:\n{sample.states[s]}")
            return ExitCode.SeparationPolicyNotComplete

    return ExitCode.Success

import logging
import os

from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io import FstripsReader
from tarski.search import GroundForwardSearchModel
from tarski.search.applicability import is_applicable
from tarski.search.blind import make_child_node, make_root_node, SearchStats
from tarski.search.model import progress
from tarski.syntax.transform.action_grounding import ground_schema_into_plain_operator_from_grounding

from . import cnfgen
from .driver import Step, BENCHMARK_DIR
from .featuregen import generate_feature_pool
from .features import create_model_factory, compute_static_atoms, generate_model_from_state
from .outputs import print_transition_matrix
from .returncodes import ExitCode
from .sampling import log_sampled_states, TransitionSample, mark_optimal_transitions
from .util.command import execute
from .util.naming import compute_sample_filenames, compute_info_filename, compute_maxsat_filename

from collections import OrderedDict


class IncrementalPolicyGenerationSingleStep(Step):
    """ Generate exhaustively a set of all features up to a given complexity from the transition (state) sample """

    def get_required_attributes(self):
        return []

    def get_required_data(self):
        return []

    def process_config(self, config):
        return config

    def description(self):
        return "Incremental policy generation step"

    def get_step_runner(self):
        return run


def compute_policy_for_sample(config, sample):
    code, res = cnfgen.run(config, None, None)
    return None if code != ExitCode.Success else res['d2l_policy']


def run_fd(domain, instance):
    """ Run Fast Downward on the given problem, and return a plan, or None if
    the problem is not solvable. """
    # e.g. fast-downward.py --alias seq-opt-lmcut /home/frances/projects/code/downward-benchmarks/gripper/prob01.pddl
    logging.info(f'Invoking Fast Downward on instance {instance}')
    args = f'--alias seq-opt-lmcut --plan-file plan.txt {domain} {instance}'.split()
    retcode = execute(['fast-downward.py'] + args)
    if retcode != 0:
        logging.error("Fast Downward error")
        return None

    with open('plan.txt', 'r') as f:
        # Read up all lines in plan file that do not start with a comment character ";"
        plan = [line for line in f.read().splitlines() if not line.startswith(';')]
    return plan


def generate_plan_and_create_sample(domain, instance_id, instance, sample):
    problem = parse_problem_with_tarski(domain, instance)
    model = GroundForwardSearchModel(problem, ground_problem_schemas_into_plain_operators(problem))

    # Run Fast Downward and get a plan!
    plan = run_fd(domain, instance)
    if plan is None:
        # instance is unsolvable
        print("Unsolvable instance")
        return

    # Collect all states in the plan
    allstates = []
    s = problem.init
    for action in plan:
        allstates.append(s)
        components = action.rstrip(')').lstrip('(').split()
        assert len(components) > 0
        a = problem.get_action(components[0])
        op = ground_schema_into_plain_operator_from_grounding(a, components[1:])
        if not is_applicable(s, op):
            raise RuntimeError(f"Action {op} from FD plan not applicable!")

        s = progress(s, op)
    # Note that we don't need to add the last state, as it's been already added as a child of its parent:
    # sample_writer.add_state(s, expand=False)

    if not model.is_goal(s):
        raise RuntimeError(f"State after executing FD plan is not a goal: {s}")

    # Finally, add the states (plus their expansions) to the D2L Transitionsample object.
    for i, state in enumerate(allstates, start=0):
        expand_state_into_sample(sample, state, instance_id, model, root=(i == 0))


def expand_state_into_sample(sample, state, instance_id, model, root=False):
    sid = sample.add_state(state, instance_id=instance_id, expanded=True, goal=model.is_goal(state),
                           unsolvable=False, root=root)

    for op, sprime in model.successors(state):
        # Note that we set update_if_duplicate=False to prevent this from updating a previously seen
        # state that is in the plan trace.
        succid = sample.add_state(sprime, instance_id=instance_id, expanded=False, goal=model.is_goal(sprime),
                                  unsolvable=False, root=False, update_if_duplicate=False)
        sample.add_transition(sid, succid)


def generate_initial_sample(config):
    # To generate the initial sample, we compute one plan per training instance, and include in the sample all
    # states that are part of the plan, plus all their (possibly unexpanded) children.
    # _run_brfs(config, None, None)

    sample = TransitionSample()
    for i, instance in enumerate(config.instances, start=0):
        generate_plan_and_create_sample(config.domain, i, instance, sample)

    mark_optimal_transitions(sample)
    logging.info(f"Entire sample: {sample.info()}")

    # Since we have only generated some (optimal) plan, we don't know the actual V* value for states that are in the
    # sample but not in the plan. We mark that accordingly with the special -2 "unknown" value
    for s in sample.states.keys():
        if s not in sample.expanded and s not in sample.goals:
            sample.vstar[s] = -2

    print_transition_matrix(sample, config.transitions_info_filename)
    log_sampled_states(sample, config.resampled_states_filename)

    return sample


def parse_problem_with_tarski(domain_file, inst_file):
    reader = FstripsReader(raise_on_error=True, theories=None, strict_with_requirements=False, case_insensitive=True)
    return reader.read_problem(domain_file, inst_file)


class SafePolicyGuidedSearch:
    """ Apply a given policy (mapping states to actions) to the underlying search model.
    The search is "safe" in the sense that it detects duplicate nodes, i.e. loops induced by the policy,
    at the cost of keeping a closed list whose size grows with the size of the state space.
    """

    def __init__(self, model: GroundForwardSearchModel, policy):
        self.model = model
        self.policy = policy

    def run(self):
        return self.search(self.model.init())

    def search(self, root):
        """ Run the policy-guided search.
        When the policy is not defined in some state, return that state.
        When the policy enters a loop, return the entire loop.
        """
        stats = SearchStats()

        closed = {root}
        current = make_root_node(root)

        while True:
            # print(current.state)
            stats.iterations += 1
            # logging.debug(f"brfs: Iteration {iteration}")

            if self.model.is_goal(current.state):
                logging.info(f"Goal found after {stats.nexpansions} expansions")
                return True, {current.state}

            child, operator = self.policy(current.state)
            if operator is None:
                logging.error(f"Policy not defined on state {current.state}")
                return False, {current.state}

            current = make_child_node(current, operator, child)
            if current.state in closed:
                loop = retrieve_loop_states(current)
                logging.error(f"Size-{len(loop)} loop detected after {len(closed)} expansions. State: {current.state}")
                return False, loop

            closed.add(current.state)
            stats.nexpansions += 1


def retrieve_loop_states(node):
    """ Retrieve the states that are part of a loopy trajectory within the state space of the problem. """
    assert node.parent is not None  # The root cannot be a loop
    loop = {node.state}
    x = node.parent
    while x.state != node.state:
        loop.add(x.state)
        x = x.parent

    return loop


class D2LPolicy:
    def __init__(self, search_model, tc_policy, model_factory, static_atoms):
        self.search_model = search_model
        self.tc_policy = tc_policy
        self.model_factory = model_factory
        self.static_atoms = static_atoms

    def __call__(self, state):
        m0 = generate_model_from_state(self.model_factory, state, self.static_atoms)
        for operator, succ in self.search_model.successors(state):
            m1 = generate_model_from_state(self.model_factory, succ, self.static_atoms)
            if self.tc_policy.transition_is_good(m0, m1):
                return succ, operator
        return None, None


def tarski_model_to_d2l_sample_representation(state):
    atoms = list()
    for atom in state.as_atoms():
        atoms.append([atom.predicate.name] + [str(st) for st in atom.subterms])
    return tuple(sorted(atoms))


def test_policy_and_compute_flaws(policy, instances, config, sample=None):
    """
    If sample is not None, the computed flaws are added to it.
    """
    nsolved = 0
    for instance_id, instance in enumerate(instances):
        # Parse the domain & instance and create a model generator
        problem, dl_model_factory = create_model_factory(config.domain, instance, config.parameter_generator)
        static_atoms, _ = compute_static_atoms(problem)

        search_model = GroundForwardSearchModel(problem, ground_problem_schemas_into_plain_operators(problem))
        d2l_policy = D2LPolicy(search_model, policy, dl_model_factory, static_atoms)
        search = SafePolicyGuidedSearch(search_model, d2l_policy)

        # Collect all the states from which we want to test the policy
        roots = {problem.init}
        if config.refine_policy_from_entire_sample and sample is not None:
            roots.update(sample.get_leaves())

        testruns = [search.search(root) for root in roots]
        if all(res is True for res, _ in testruns):
            print("Policy solves instance")
            nsolved += 1
            continue

        # result, visited_states = search.search(root)
        flaws = set().union(*(flaws for res, flaws in testruns if res is not True))

        # If a sample was provided, add the found flaws to it
        if sample is not None:
            for state in flaws:
                sid = sample.get_state_id(state)
                if sid is not None and sample.is_expanded(sid):
                    # If the state is already in the sample and already expanded, no need to do anything about it
                    continue

                expand_state_into_sample(sample, state, instance_id, search_model, root=False)

    return nsolved


def run(config, data, rng):
    config.sample_files = compute_sample_filenames(**config.__dict__)
    config.test_sample_files = []
    config.resampled_states_filename = compute_info_filename(config.__dict__, 'sample.txt')
    config.transitions_info_filename = compute_info_filename(config.__dict__, "transitions-info.io")
    config.concept_denotation_filename = compute_info_filename(config.__dict__, "concept-denotations.txt")
    config.feature_denotation_filename = compute_info_filename(config.__dict__, "feature-denotations.txt")
    config.serialized_feature_filename = compute_info_filename(config.__dict__, "serialized-features.io")
    config.top_filename = compute_info_filename(config.__dict__, "top.dat")
    config.cnf_filename = compute_maxsat_filename(config.__dict__)
    config.good_transitions_filename = compute_info_filename(config.__dict__, "good_transitions.io")
    config.good_features_filename = compute_info_filename(config.__dict__, "good_features.io")
    config.wsat_varmap_filename = compute_info_filename(config.__dict__, "varmap.wsat")
    config.wsat_allvars_filename = compute_info_filename(config.__dict__, "allvars.wsat")
    #config.refine_policy_from_entire_sample=True

    config.validation_instances = [os.path.join(BENCHMARK_DIR, config.domain_dir, i) for i in
                                   config.validation_instances]

    # Compute a plan for each of the training instances, and put all states in the plan into the sample, along with
    # all of their (possibly non-expanded) children.
    # All states will need to be labeled with their status (goal / unsolvable / alive)
    sample = generate_initial_sample(config)
    iteration = 1
    while True:
        # TODO: This could be optimized to avoid recomputing the features over the states that already were in the
        #       sample on the previous iteration.
        print(f"### MAIN ITERATION: {iteration}; SAMPLE SIZE: {sample.num_states()}")
        code, res = generate_feature_pool(config, sample)
        assert code == ExitCode.Success

        policy = compute_policy_for_sample(config, sample)
        if not policy:
            print("No policy found under given complexity bound")
            break

        # Test the policy on the validation set
        nsolved = test_policy_and_compute_flaws(policy, config.validation_instances, config, sample)
        if nsolved == len(config.validation_instances):
            print("Policy solves all states in training set")
            break  # Policy test was successful, we're done.

        log_sampled_states(sample, config.resampled_states_filename)
        print_transition_matrix(sample, config.transitions_info_filename)

        iteration += 1

    # Run on the test set and report coverage. No need to add flaws to sample here
    if policy:
        nsolved = test_policy_and_compute_flaws(policy, config.test_policy_instances, config)
        print(f"Learnt policy solves {nsolved}/{len(config.test_policy_instances)} in the test set")
    return ExitCode.Success, {}

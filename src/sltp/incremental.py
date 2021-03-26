import logging
import os

from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io import FstripsReader
from tarski.search import GroundForwardSearchModel, BreadthFirstSearch
from tarski.search.applicability import is_applicable
from tarski.search.blind import make_child_node, make_root_node, SearchSpace, SearchStats
from tarski.search.model import progress
from tarski.syntax.transform.action_grounding import ground_schema_into_plain_operator_from_grounding

from . import cnfgen
from .driver import Step, BENCHMARK_DIR
from .featuregen import generate_feature_pool
from .features import FeatureInterpreter, create_model_factory, compute_static_atoms, generate_model_from_state
from .outputs import print_transition_matrix
from .returncodes import ExitCode
from .sampling import generate_sample, sample_generated_states, log_sampled_states
from .steps import _run_brfs
from .util.command import execute
from .util.naming import compute_sample_filenames, compute_info_filename, compute_maxsat_filename


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


class TarskiSampleWriter:
    """ A simple class to manage the printing of the state space samples that get explored by the Tarski engine """
    def __init__(self, search_model, outfile):
        self.outfile = outfile
        self.search_model = search_model
        self.states = dict()
        self.expanded = set()

    def register_state(self, state, state_string):
        sid = len(self.states)
        self.states[state_string] = sid

        is_blind = False  # I think this is not being used at the moment
        print(f"(N) {sid} {int(self.search_model.is_goal(state))} {int(is_blind)} {' '.join(state_string)}", file=self.outfile)

        return sid

    def add_state(self, state, expand=True):
        asstr = translate_state(state, fmt="std")
        sid = self.states.get(asstr, None)

        if sid is not None and sid in self.expanded:
            # The state has already been expanded, no need to add it again to the sample file
            return

        sid = self.register_state(state, asstr) if sid is None else sid

        self.expanded.add(sid)

        for op, sprime in self.search_model.successors(state):
            succstr = translate_state(sprime, fmt="std")
            succid = self.states.get(succstr, None)
            if succid is None:
                succid = self.register_state(sprime, succstr)
            print(f"(E) {sid} {succid}", file=self.outfile)


def generate_plan_and_create_sample(domain, instance, outfile):
    problem = parse_problem_with_tarski(domain, instance)
    model = GroundForwardSearchModel(problem, ground_problem_schemas_into_plain_operators(problem))

    with open(outfile, 'w') as f:
        logging.info(f"Printing plan sample info to file {outfile}")
        sample_writer = TarskiSampleWriter(model, f)

        plan = run_fd(domain, instance)
        if plan is None:
            # instance is unsolvable
            print("Unsolvable instance")
            return

        s = problem.init
        for action in plan:
            sample_writer.add_state(s)
            components = action.rstrip(')').lstrip('(').split()
            assert len(components) > 0
            a = problem.get_action(components[0])
            op = ground_schema_into_plain_operator_from_grounding(a, components[1:])
            if not is_applicable(s, op):
                raise RuntimeError(f"Action {op} from FD plan not applicable!")

            s = progress(s, op)
        # sample_writer.add_state(s, expand=False)  # Note that we don't need to add the last state, as it's been already added as a child of its parent

        if not model.is_goal(s):
            raise RuntimeError(f"State after executing FD plan is not a goal: {s}")


def generate_initial_sample(config):

    # To generate the initial sample, we compute one plan per training instance, and include in the sample all
    # states that are part of the plan, plus all their (possibly unexpanded) children.
    # _run_brfs(config, None, None)

    for instance, outfile in zip(config.instances, config.sample_files):
        generate_plan_and_create_sample(config.domain, instance, outfile)

    sample = sample_generated_states(config, None)

    # Since we have only generated some (optimal) plan, we don't know the actual V* value for states that are in the
    # sample but not in the plan. We mark that accordingly with the special -2 "unknown" value
    for s in sample.states.keys():
        if s not in sample.expanded and s not in sample.goals:
            sample.vstar[s] = -2

    log_sampled_states(sample, config.resampled_states_filename)
    print_transition_matrix(sample, config.transitions_info_filename)

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
        # create obj to track state space
        space = SearchSpace()
        stats = SearchStats()

        closed = {root}
        current = make_root_node(root)

        while True:
            stats.iterations += 1
            # logging.debug(f"brfs: Iteration {iteration}")

            if self.model.is_goal(current.state):
                logging.info(f"Goal found after {stats.nexpansions} expansions")
                return True, current

            child, operator = self.policy(current.state)
            if operator is None:
                logging.error(f"Policy not defined on state {current.state}")
                return False, current.state

            current = make_child_node(current, operator, child)
            if current.state in closed:
                logging.error(f"Loop detected in state {current.state}")
                return False, current.state, None, None

            closed.add(current.state)
            stats.nexpansions += 1


def translate_atom(atom, fmt="lisp"):
    if fmt != "lisp":
        return str(atom)

    if not atom.subterms:
        return f"({atom.predicate.name})"

    args = ' '.join(str(a) for a in atom.subterms)
    return f"({atom.predicate.name} {args})"


def translate_state(state, fmt="lisp"):
    """ Translate a Tarski state into a list of strings, one for each atom that is true in the state. """
    return tuple(sorted(translate_atom(a, fmt=fmt) for a in state.as_atoms()))


class D2LPolicy:
    def __init__(self, search_model, tc_policy, model_factory, static_atoms):
        self.search_model = search_model
        self.tc_policy = tc_policy
        self.model_factory = model_factory
        self.static_atoms = static_atoms
    
    def __call__(self, state):
        m0 = generate_model_from_state(self.model_factory, translate_state(state), self.static_atoms)
        for operator, succ in self.search_model.successors(state):
            m1 = generate_model_from_state(self.model_factory, translate_state(succ), self.static_atoms)
            if self.tc_policy.transition_is_good(m0, m1):
                return succ, operator
        return None, None


def test_policy(policy, instances, config):
    flaws = []
    solved = 0
    for instance in instances:
        # Parse the domain & instance and create a model generator
        problem, dl_model_factory = create_model_factory(config.domain, instance, config.parameter_generator)
        static_atoms, _ = compute_static_atoms(problem)
        search_model = GroundForwardSearchModel(problem, ground_problem_schemas_into_plain_operators(problem))
        
        d2l_pol = D2LPolicy(search_model, policy, dl_model_factory, static_atoms)
        search = SafePolicyGuidedSearch(search_model, d2l_pol)
        result, state = search.run()
        if result:
            print("Policy solves instance")
            solved += 1
            continue
        flaws.append(state)
    return flaws, solved


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

    config.validation_instances = [os.path.join(BENCHMARK_DIR, config.domain_dir, i) for i in config.validation_instances]

    # Compute a plan for each of the training instances, and put all states in the plan into the sample, along with
    # all of their (possibly non-expanded) children.
    # All states will need to be labeled with their status (goal / unsolvable / alive)
    sample = generate_initial_sample(config)

    while True:
        # TODO: This could be optimized to avoid recomputing the features over the states that already were in the
        #       sample on the previous iteration.
        code, res = generate_feature_pool(config, sample)
        assert code == ExitCode.Success

        policy = compute_policy_for_sample(config, sample)
        if not policy:
            print("No policy found under given complexity bound")
            break

        # Test the policy on the validation set
        flaws, _ = test_policy(policy, config.validation_instances, config)
        if not flaws:
            print("Policy solves all tests")
            break  # Policy test was successful, we're done.

        # Add the flaws found during validation to the sample
        sample = sample.add(flaws)

    # Run on the test set and report coverage.
    if policy:
        _, nsolved = test_policy(policy, config.test_policy_instances, config)
        print(f"Learnt policy solves {nsolved}/{len(config.test_policy_instances)} in the test set")
    return ExitCode.Success, {}



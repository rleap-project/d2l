import logging
import os

from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io import FstripsReader
from tarski.search import GroundForwardSearchModel, BreadthFirstSearch
from tarski.search.blind import make_child_node, make_root_node, SearchSpace, SearchStats

from . import cnfgen
from .driver import Step, BENCHMARK_DIR
from .featuregen import generate_feature_pool
from .features import FeatureInterpreter, create_model_factory, compute_static_atoms, generate_model_from_state
from .returncodes import ExitCode
from .sampling import generate_sample
from .steps import _run_brfs
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


def generate_initial_sample(config):
    # ground_actions = ground_problem_schemas_into_plain_operators(problem)
    # model = GroundForwardSearchModel(problem, ground_actions)
    #
    # search = BreadthFirstSearch(model)
    # space, stats = search.run()

    # TODO Run Fast Downward on the instance(s), retrieve the plans, and put them
    # TODO into the sample. Since we don't have that yet, at the moment we run a simple brfs:
    _run_brfs(config, None, None)
    sample = generate_sample(config, None)
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


def translate_atom(atom):
    if not atom.subterms:
        return f"({atom.predicate.name})"
    args = ' '.join(str(a) for a in atom.subterms)
    return f"({atom.predicate.name} {args})"


class D2LPolicy:
    def __init__(self, search_model, tc_policy, model_factory, static_atoms):
        self.search_model = search_model
        self.tc_policy = tc_policy
        self.model_factory = model_factory
        self.static_atoms = static_atoms
    
    def __call__(self, state):
        strrep = [translate_atom(a) for a in state.as_atoms()]
        m0 = generate_model_from_state(self.model_factory, strrep, self.static_atoms)
        for operator, succ in self.search_model.successors(state):
            strrep = [translate_atom(a) for a in succ.as_atoms()]
            m1 = generate_model_from_state(self.model_factory, strrep, self.static_atoms)
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
        # This could be optimized to avoid recomputing the features over the states that already were in the sample
        # on the previous iteration.
        code, res = generate_feature_pool(config, sample)

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
    _, nsolved = test_policy(policy, config.test_policy_instances, config)
    print(f"Learnt policy solves {nsolved}/{len(config.test_policy_instances)} in the test set")
    return ExitCode.Success, {}



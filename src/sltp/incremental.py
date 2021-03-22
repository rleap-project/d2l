from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io import FstripsReader
from tarski.search import GroundForwardSearchModel, BreadthFirstSearch

from . import cnfgen
from .driver import Step
from .featuregen import generate_feature_pool
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
    return None if code == ExitCode.Success else res['d2l_policy']


def generate_initial_sample(config):
    reader = FstripsReader(raise_on_error=True, theories=[])
    problem = reader.read_problem(config.domain, config.instances[0])

    ground_actions = ground_problem_schemas_into_plain_operators(problem)
    model = GroundForwardSearchModel(problem, ground_actions)

    search = BreadthFirstSearch(model)
    space, stats = search.run()


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

    # Compute a plan for each of the training instances, and put all states in the plan into the sample, along with
    # all of their (possibly non-expanded) children.
    # All states will need to be labeled with their status (goal / unsolvable / alive)
    sample = generate_initial_sample(config)

    while True:
        # This could be optimized to avoid recomputing the features over the states that already were in the sample
        # on the previous iteration.
        code, res = generate_feature_pool(config, sample)

        # _run_brfs(config, data, rng)
        # sample = generate_sample(config, rng)
        policy = compute_policy_for_sample(config, sample)
        if not policy:
            print("No policy found under given complexity bound")
            break

        # Use pyperplan to test the policy on the validation set
        flaws = test_policy(policy)
        if not flaws:
            print("Policy solves all tests")
            break  # Policy test was successful, we're done.

        # Add the flaws found during validation to the sample
        sample = sample.add(flaws)

    print("Done!")
    # Run on the test set and report coverage.
    report_policy_coverage(policy)



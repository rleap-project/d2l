import os
from enum import Enum

from .command import create_experiment_workspace
from .. import SLTP_SRC_DIR
from ..driver import Experiment, BENCHMARK_DIR, BASEDIR
from ..language import parse_pddl
from ..steps import generate_pipeline


class OptimizationPolicy(Enum):
    NUM_FEATURES = 1  # Minimize number of features
    TOTAL_FEATURE_COMPLEXITY = 2  # Minimize the sum of depths of selected features
    NUMERIC_FEATURES_FIRST = 3   # Minimize number of numeric features first, then overall features.


def get_experiment_class(kwargs):
    exptype = kwargs.get('experiment_type', None)
    if exptype is not None:  # parameter 'experiment_type' takes precedence
        return Experiment

    expclass = kwargs.get('experiment_class', None)
    return expclass if expclass is not None else Experiment


def generate_experiment(expid, domain_dir, domain, **kwargs):
    """ """

    if "instances" not in kwargs:
        raise RuntimeError("Please specify domain and instance when generating an experiment")

    instances = kwargs["instances"] if isinstance(kwargs["instances"], (list, tuple)) else [kwargs["instances"]]
    kwargs["domain"] = os.path.join(BENCHMARK_DIR, domain_dir, domain)
    kwargs["instances"] = [os.path.join(BENCHMARK_DIR, domain_dir, i) for i in instances]

    if "test_domain" in kwargs:
        kwargs["test_domain"] = os.path.join(BENCHMARK_DIR, domain_dir, kwargs["test_domain"])
        kwargs["test_instances"] = \
            [os.path.join(BENCHMARK_DIR, domain_dir, i) for i in kwargs.get("test_instances", [])]
        kwargs["test_policy_instances"] = \
            [os.path.join(BENCHMARK_DIR, domain_dir, i) for i in kwargs.get("test_policy_instances", [])]
    else:
        kwargs["test_domain"] = None
        kwargs["test_instances"] = []
        kwargs["test_policy_instances"] = []

    defaults = dict(
        pipeline="d2l_pipeline",

        # Some directories of external tools needed by the pipeline
        generators_path=os.path.join(os.path.dirname(SLTP_SRC_DIR), "generators"),
        pyperplan_path=os.path.join(os.path.dirname(SLTP_SRC_DIR), "pyperplan"),

        # The directory where the experiment outputs will be left
        workspace=os.path.join(BASEDIR, 'workspace'),

        # Location of the FS planner, used to do the state space sampling
        planner_location=os.getenv("FS_PATH", os.path.expanduser("~/projects/code/fs")),

        # Type of sampling procedure. Only breadth-first search implemented ATM
        driver="bfs",

        # Type of sampling. Accepted options are:
        # - "all" (default): Use all expanded states
        # - "random": Use only a random sample of the expanded states, of size given by the option "num_sampled_states"
        # - "optimal": Use those expanded states on some optimal path (only one arbitrary optimal path)
        # Note: ATM random sampling is deactivated
        sampling="all",

        # Number of states to be expanded in the sampling procedure. Either a positive integer, or the string
        # "until_first_goal", or the string "all", both with obvious meanings
        num_states="all",

        # Number randomly sampled states among the set of expanded states. The default of None means
        # all expanded states will be used
        num_sampled_states=None,

        # Max. size of the generated concepts
        max_concept_size=10,

        # Provide a special, handcrafted method to generate concepts, if desired.
        # This will override the standard concept generation procedure (default: None)
        concept_generator=None,

        # Or, alternatively, provide directly the features instead of the concepts (default: None)
        feature_generator=None,

        # Max. allowed complexity for distance and conditional features (default: 0)
        distance_feature_max_complexity=0,
        cond_feature_max_complexity=0,

        # Whether to generate comparison features of the type F1 < F2
        comparison_features=False,

        # Method to generate domain parameters (goal or otherwise). If None, goal predicates will
        # be used (default: None)
        parameter_generator=None,

        # Whether to create goal-identifying features (e.g. of the form p_g AND not p_s for every unary predicate
        # apperaring in the goal)
        create_goal_features_automatically=False,

        # Optionally, use a method that gives handcrafted names to the features
        # (default: None, which will use their string representation)
        feature_namer=default_feature_namer,

        # What optimization criteria to use in the max-sat problem
        optimization=OptimizationPolicy.TOTAL_FEATURE_COMPLEXITY,

        # the max-sat solver to use. Accepted: openwbo, openwbo-inc, wpm3, maxino
        maxsat_solver='openwbo',
        maxsat_timeout=None,

        domain_dir=domain_dir,

        # The Experiment class to be used (e.g. standard, or incremental)
        experiment_class=get_experiment_class(kwargs),

        # Reduce output to a minimum
        quiet=False,

        # Number of states to expand & test on the testing instances
        num_tested_states=50000,

        # Some debugging help to print the denotations of all features over all states (expensive!)
        print_denotations=False,

        # By default don't timeout the concept generation process
        concept_generation_timeout=-1,

        # A function to manually provide a transition-classification policy that we want to test
        d2l_policy=None,

        # In the transition-separation encoding, whether we want to exploit the equivalence relation
        # among transitions given by the feature pool
        use_equivalence_classes=False,

        # In the transition-separation encoding, whether we want to exploit the dominance among features to ignore
        # dominated features and reduce the size of the encoding
        use_feature_dominance=False,

        # Whether to automatically generate goal-distinguishing concepts and roles
        generate_goal_concepts=False,

        # The slack value for the maximum allowed value for V_pi(s) = slack * V^*(s)
        v_slack=2,

        # The number of features of the policy graph abstraction
        n_features=3,

        # In the transition-separation encoding, whether to post constraints to ensure distinguishability of goals
        distinguish_goals=False,

        # A function to create the FOL language, used to be able to parse the features.
        language_creator=pddl_language_creator,

        # Set a random seed for reproducibility (default: 1)
        seed=123,

        # The hyper-parameters of the incremental approach
        initial_sample_size=50,
        refinement_batch_size=20,

        # The level of verbosity - higher value means more verbosity
        verbosity=0,

        # The type of acyclicity encoding to be used: "topological", "asp", "sd2l".
        acyclicity="topological",

        encodings_dir=os.path.join(os.path.dirname(SLTP_SRC_DIR), "encodings"),
        sampling_strategy="random",

        # In the SD2L encoding, whether to enforce policy-closedness constraints too
        closed=True,

        # Print hstar value as a feature matrix column.
        print_hstar_in_feature_matrix=False,

        # Generate DL disjunctions
        generate_or_concepts=False,

        # Interpret infinity values of features as zero
        infty_feat_as_zero=False,

        # ???
        optimal_steps=0,
        consistency_bound=10,

        refine_policy_from_entire_sample=True,

        # Compute a plan from every detected policy flaw
        compute_plan_on_flaws=False,

        num_random_walks=0,

        random_walk_length=10,
    )

    parameters = {**defaults, **kwargs}  # Copy defaults, overwrite with user-specified parameters

    parameters['experiment_dir'] = os.path.join(parameters['workspace'], expid.replace(':', '_'))
    create_experiment_workspace(parameters["experiment_dir"], rm_if_existed=False)

    steps = generate_pipeline(**parameters)
    exp = parameters["experiment_class"](steps, parameters)
    return exp


def default_feature_namer(s):
    return str(s)


def pddl_language_creator(config):
    _, language, _ = parse_pddl(config.domain)
    return language


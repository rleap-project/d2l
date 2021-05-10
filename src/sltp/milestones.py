import copy
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

from tarski.dl import compute_dl_vocabulary
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io import FstripsReader, FstripsWriter
from tarski.search import GroundForwardSearchModel
from tarski.search.model import progress, is_applicable
from tarski.syntax.transform.action_grounding import ground_schema_into_plain_operator_from_grounding

from .cnfgen import invoke_cpp_module, parse_dnf_policy
from .driver import Step
from .featuregen import print_sample_info, deal_with_serialized_features, generate_output_from_handcrafted_features, \
    invoke_cpp_generator, goal_predicate_name
from .features import generate_model_from_state, \
    compute_predicates_appearing_in_goal, compute_instance_information, create_model_cache
from .language import parse_pddl
from .learners.explore import filter_features
from .learners.utils import describe
from .models import DLModelFactory
from .outputs import print_transition_matrix
from .returncodes import ExitCode
from .util.command import execute
from .util.naming import compute_sample_filenames, compute_info_filename, compute_maxsat_filename


class MilestonesFeatureGenerationStep(Step):
    """ Generate exhaustively a set of all features up to a given complexity from the transition (state) sample """
    def get_required_attributes(self):
        return []

    def get_required_data(self):
        return []

    def process_config(self, config):
        return config

    def description(self):
        return "Feature generation step for the Milestone approach"

    def get_step_runner(self):
        return generate_features


class HStarDecreasingClassifierGenerationStep(Step):
    """ Generate exhaustively a set of all features up to a given complexity from the transition (state) sample """
    def get_required_attributes(self):
        return []

    def get_required_data(self):
        return []

    def process_config(self, config):
        return config

    def description(self):
        return "Classifier computation step for the Milestone approach"

    def get_step_runner(self):
        return generate_policy


def compute_dnf_classifier(config, language):
    exitcode = invoke_cpp_module(config, None)
    if exitcode != ExitCode.Success:
        return None

    # Parse the DNF transition-classifier and transform it into a policy
    policy = parse_dnf_policy(config, language)

    policy.minimize()
    print("Final Policy:")
    policy.print_aaai20()

    return policy


def run_fd(config, domain, instance, verbose):
    """ Run Fast Downward on the given problem, and return a plan, or None if
    the problem is not solvable. """
    # e.g. fast-downward.py --alias seq-opt-lmcut /home/frances/projects/code/downward-benchmarks/gripper/prob01.pddl

    exp_dir = config.__dict__["experiment_dir"]
    with tempfile.NamedTemporaryFile(mode='w', delete=True) as tf:
        args = f'--alias seq-opt-lmcut --plan-file {exp_dir}/plan.txt {domain} {instance}'.split()
        stdout = None if verbose else tf.name
        retcode = execute(['fast-downward.py'] + args, stdout=stdout)
        if retcode in (11, 12):
            return None
        elif retcode != 0:
            raise RuntimeError(f"Fast Downward exited with unexpected code {retcode}")

    with open(f'{exp_dir}/plan.txt', 'r') as f:
        # Read up all lines in plan file that do not start with a comment character ";"
        plan = [line for line in f.read().splitlines() if not line.startswith(';')]
    return plan


def generate_instance_file(problem, pddl_constants):
    writer = FstripsWriter(problem)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
        tf.write(writer.print_instance(pddl_constants))
    return tf.name


def compute_plan(config, model, domain_filename, instance_filename, init):
    # Run Fast Downward and get a plan!
    plan = run_fd(config, domain_filename, instance_filename, verbose=False)
    if plan is None:
        # instance is unsolvable
        # print("Unsolvable instance")
        return None

    # Collect all states in the plan
    allstates = []
    s = init
    for action in plan:
        allstates.append(s)
        components = action.rstrip(')').lstrip('(').split()
        assert len(components) > 0
        a = model.problem.get_action(components[0])
        op = ground_schema_into_plain_operator_from_grounding(a, components[1:])
        if not is_applicable(s, op):
            raise RuntimeError(f"Action {op} from FD plan not applicable!")

        s = progress(s, op)

    if not model.is_goal(s):
        raise RuntimeError(f"State after executing FD plan is not a goal: {s}")

    allstates.append(s)
    return allstates


def random_walk(rng, model, length):
    s = model.problem.init
    for _ in range(0, length):
        s = choose_random_successor(rng, model, s)
    return s


def choose_random_successor(rng, model, s):
    succs = [succ for _, succ in model.successors(s)]
    return rng.choice(succs) if succs else None


class SimpleSample:
    def __init__(self):
        self.states = []
        self.state_to_id = dict()
        self.state_to_hstar = dict()
        self.transitions = set()
        self.instance = dict()  # A mapping between states and the problem instances they came from

    def add_state(self, state, instance, hstar):
        sid = self.state_to_id.get(state)
        if sid is not None:
            return sid
        sid = self.state_to_id[state] = len(self.states)
        self.states.append(state)
        self.state_to_hstar[state] = hstar
        self.instance[sid] = instance
        return sid

    def in_sample(self, state):
        return state in self.state_to_id

    def get_id(self, state):
        return self.state_to_id[state]

    def add_transition(self, sid, sprime_id, polarity):
        self.transitions.add((sid, sprime_id, polarity))

    def info(self):
        return f"#states: {len(self.states)}; #transitions: {len(self.transitions)}"

    def __str__(self):
        return f"SimpleSample[{self.info()}]"


class SampleGenerator:
    def __init__(self, config, rng, training_instances_data):
        self.rng = rng
        self.config = config
        self.domain_filename = config.domain
        self.training_instances_data = training_instances_data
        self.sample = SimpleSample()
        self.planner_runs = 0
        self.unsolvable_hit = 0
        self.hstar_cache = dict()

    def compute_plan(self, config, model, pddl_constants, state):
        prob = copy.copy(model.problem)
        prob.init = state
        instance_filename = generate_instance_file(prob, pddl_constants)
        self.planner_runs += 1
        plan = compute_plan(config, model, self.domain_filename, instance_filename, state)
        if plan is None:
            self.unsolvable_hit += 1
        return plan

    def compute_hstar_uncached(self, model, instance_data, state):
        plan_states = self.compute_plan(self.config, model, instance_data['pddl_constants'], state)
        if plan_states is None:
            self.sample.add_state(state, instance_data['id'], sys.maxsize)
            return sys.maxsize

        previous = None
        for d, state in enumerate(reversed(plan_states), start=0):
            sid = self.sample.add_state(state, instance_data['id'], d)
            if d > 0:
                # By definition, all transitions in an optimal plan are h*-decreasing
                self.sample.add_transition(sid, previous, True)
            previous = sid
        return len(plan_states)-1

    def compute_hstar(self, model, instance_data, state):
        hstar = self.hstar_cache.get(state)
        if hstar is not None:
            return hstar

        hstar = self.hstar_cache[state] = self.compute_hstar_uncached(model, instance_data, state)
        return hstar

    def add_point(self, sid, succ_id, positive):
        self.sample.add_transition(sid, succ_id, positive)

    def generate(self):
        for instance in self.config.instances:
            instance_data = self.training_instances_data[instance]
            model = instance_data['search_model']

            for _ in range(0, self.config.num_random_rollouts):
                s = model.init()

                self.process(instance_data, model, s)
                for _ in range(0, self.config.random_walk_length):
                    s = choose_random_successor(self.rng, model, s)
                    if s is None:
                        break
                    self.process(instance_data, model, s)

        print_transition_matrix(self.sample, self.config.transitions_info_filename)
        # log_sampled_states(self.sample, self.config.resampled_states_filename)
        print(f"Sample generated with {self.planner_runs} FD runs, of which"
              f" {self.unsolvable_hit} on an unsolvable instance")
        return self.sample

    def process(self, instance_data, model, s):
        succs_hstar = [(self.compute_hstar(model, instance_data, sprime), sprime) for op, sprime in model.successors(s)]
        hstar = min(h for h, _ in succs_hstar) + 1 if succs_hstar else sys.maxsize
        self.sample.add_state(s, instance_data['id'], hstar)
        for hsucc, succ in succs_hstar:
            self.add_point(self.sample.get_id(s), self.sample.get_id(succ), hsucc < hstar)


def parse_problem_with_tarski(domain_file, inst_file):
    reader = FstripsReader(raise_on_error=True, theories=None, strict_with_requirements=False, case_insensitive=True)
    return reader.read_problem(domain_file, inst_file)


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


def generate_feature_pool(config, sample, all_instance_data):
    logging.info(f"Starting generation of feature pool. State sample used to detect redundancies: {sample.info()}")

    all_objects = []
    all_predicates, all_functions = set(), set()
    goal_predicate_info = set()

    infos = []
    for instance_name in config.instances:
        instance_data = all_instance_data[instance_name]
        lang = instance_data['language']
        info = instance_data['info']
        all_objects.append(set(c.symbol for c in lang.constants()))
        all_predicates.update((p.name, p.arity) for p in lang.predicates if not p.builtin)
        all_functions.update((p.name, p.arity) for p in lang.functions if not p.builtin)

        # Add goal predicates and functions
        goal_predicate_info.update((goal_predicate_name(p.name), p.uniform_arity())
                                  for p in lang.predicates if not p.builtin and p.name in info.goal_predicates)
        goal_predicate_info.update((goal_predicate_name(f.name), f.uniform_arity())
                                   for f in lang.functions if not f.builtin and f.name in info.goal_predicates)
        all_predicates.update(goal_predicate_info)

        # Add type predicates
        all_predicates.update((p.name, 1) for p in lang.sorts if not p.builtin and p != lang.Object)

        nominals = instance_data['nominals']
        infos.append(info)

    # We assume all problems languages are the same and simply pick the last one
    vocabulary = compute_dl_vocabulary(lang)
    model_cache = create_model_cache(vocabulary, sample.states, sample.instance, nominals, infos)

    # Write out all input data for the C++ feature generator code
    print_sample_info(sample, infos, model_cache, all_predicates, all_functions, nominals,
                      all_objects, goal_predicate_info, config)

    # If user provides handcrafted features, no need to go further than here
    if config.feature_generator is not None:
        features = deal_with_serialized_features(lang, config.feature_generator, config.serialized_feature_filename)
        generate_output_from_handcrafted_features(sample, config, features, model_cache)
        return ExitCode.Success, dict(enforced_feature_idxs=[], model_cache=model_cache)

    if invoke_cpp_generator(config) != 0:
        return ExitCode.FeatureGenerationUnknownError, dict()

    return ExitCode.Success, dict(enforced_feature_idxs=[], model_cache=model_cache)


def compute_python_classifier(config, language):
    data = describe(str(Path(config.experiment_dir) / "feature-matrix.io"))

    # if args.unique_features:
    #   data = filter_duplicate_features(data)
    # if args.unique_states:
    #   data = filter_duplicate_states(data)

    maxk = max(config.distance_feature_max_complexity, config.max_concept_size)
    for k in range(2, maxk+1):
        data_k = filter_features(data, max_k=k)


def setup_config(config):
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

    #config.validation_instances = [os.path.join(BENCHMARK_DIR, config.domain_dir, i) for i in
    #                               config.validation_instances]
    return config


def generate_features(config, data, rng):
    config = setup_config(config)
    rng = np.random.default_rng(config.seed)

    # Check if user has provided some policy to be tested
    # user_policy = None if config.d2l_policy is None else generate_user_provided_policy(config)

    all_instance_data, language = compute_instance_data(config.instances, config)

    # Compute a plan for each of the training instances, and put all states in the plan into the sample, along with
    # all of their (possibly non-expanded) children.
    # All states will need to be labeled with their status (goal / unsolvable / alive)
    generator = SampleGenerator(config, rng, all_instance_data)
    sample = generator.generate()
    return generate_feature_pool(config, sample, all_instance_data)


def generate_policy(config, data, rng):
    config = setup_config(config)
    rng = np.random.default_rng(config.seed + 15)
    all_instance_data, language = compute_instance_data(config.instances, config)

    # Compute a plan for each of the training instances, and put all states in the plan into the sample, along with
    # all of their (possibly non-expanded) children.
    # All states will need to be labeled with their status (goal / unsolvable / alive)
    # generator = SampleGenerator(config, rng, all_instance_data)
    # sample = generator.generate()
    # code, res = generate_feature_pool(config, sample, all_instance_data)

    # classifier = compute_python_classifier(config, language)

    classifier = compute_dnf_classifier(config, language)
    if not classifier:
        logging.info("No classifier found under given complexity bound")
        return ExitCode.Success, {}

    # If we found a set of h*-decreasing rules, let's make sure they are correct over the training set
    sample = SampleGenerator(config, rng, all_instance_data).generate()
    print(f"Validating learnt classifier over new sample with {len(sample.transitions)} transitions")
    fps, fns = 0, 0
    for sid, succ_id, positive in sample.transitions:
        assert sample.instance[sid] == sample.instance[succ_id]
        instance_data = all_instance_data[config.instances[sample.instance[sid]]]
        dl_model_factory = instance_data['dl_model_factory']
        static_atoms = instance_data['static_atoms']

        m0 = generate_model_from_state(dl_model_factory, sample.states[sid], static_atoms)
        m1 = generate_model_from_state(dl_model_factory, sample.states[succ_id], static_atoms)
        isgood = classifier.transition_is_good(m0, m1)
        fps += isgood and not positive
        fns += not isgood and positive

    print(f"Learnt policy has {fps} false positives and {fns} false negatives out of {len(sample.transitions)} transitions")

    return ExitCode.Success, {}


def compute_instance_data(all_instances, config):
    all_instance_data = {instance: {"id": i} for i, instance in enumerate(all_instances, 0)}
    for instance_filename, data in all_instance_data.items():
        # Parse the domain & instance and create a model generator and related instance-dependent information
        problem, language, pddl_constants = parse_pddl(config.domain, instance_filename)
        nominals = pddl_constants[:]
        if config.parameter_generator is not None:
            nominals += config.parameter_generator(language)

        vocabulary = compute_dl_vocabulary(language)
        use_goal_denotation = config.parameter_generator is None
        goal_predicates = compute_predicates_appearing_in_goal(problem, use_goal_denotation)
        info = compute_instance_information(problem, goal_predicates)
        dl_model_factory = DLModelFactory(vocabulary, nominals, info)
        search_model = GroundForwardSearchModel(problem, ground_problem_schemas_into_plain_operators(problem))
        data.update(dict(language=language, problem=problem, nominals=nominals,
                         info=info, pddl_constants=pddl_constants,
                         search_model=search_model, static_atoms=info.static_atoms, dl_model_factory=dl_model_factory))
    return all_instance_data, language

import copy
import logging
import os
import tempfile
import random

from tarski.dl import compute_dl_vocabulary
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io import FstripsReader, FstripsWriter
from tarski.search import GroundForwardSearchModel
from tarski.search.applicability import is_applicable
from tarski.search.blind import make_child_node, make_root_node, SearchStats
from tarski.search.model import progress
from tarski.syntax.transform.action_grounding import ground_schema_into_plain_operator_from_grounding

from .cnfgen import invoke_cpp_module, parse_dnf_policy
from .driver import Step, BENCHMARK_DIR
from .featuregen import print_sample_info, deal_with_serialized_features, generate_output_from_handcrafted_features, \
    invoke_cpp_generator, goal_predicate_name
from .features import create_model_factory, compute_static_atoms, generate_model_from_state, report_use_goal_denotation, \
    compute_predicates_appearing_in_goal, compute_instance_information, create_model_cache
from .language import parse_pddl
from .models import DLModelFactory
from .outputs import print_transition_matrix
from .returncodes import ExitCode
from .sampling import log_sampled_states, TransitionSample, mark_optimal_transitions
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


def compute_policy_for_sample(config, language):
    exitcode = invoke_cpp_module(config, None)
    if exitcode != ExitCode.Success:
        return None

    # Parse the DNF transition-classifier and transform it into a policy
    policy = parse_dnf_policy(config, language)

    policy.minimize()
    print("Final Policy:")
    policy.print_aaai20()

    return policy


def run_fd(domain, instance, verbose):
    """ Run Fast Downward on the given problem, and return a plan, or None if
    the problem is not solvable. """
    # e.g. fast-downward.py --alias seq-opt-lmcut /home/frances/projects/code/downward-benchmarks/gripper/prob01.pddl

    with tempfile.NamedTemporaryFile(mode='w', delete=True) as tf:
        args = f'--alias seq-opt-lmcut --plan-file plan.txt {domain} {instance}'.split()
        stdout = None if verbose else tf.name
        retcode = execute(['fast-downward.py'] + args, stdout=stdout)
    if retcode != 0:
        logging.error("Fast Downward error")
        return None

    with open('plan.txt', 'r') as f:
        # Read up all lines in plan file that do not start with a comment character ";"
        plan = [line for line in f.read().splitlines() if not line.startswith(';')]
    return plan


def generate_instance_file(problem, pddl_constants):
    writer = FstripsWriter(problem)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
        tf.write(writer.print_instance(pddl_constants))
    return tf.name


def compute_plan(model, domain_filename, instance_filename, init):
    # Run Fast Downward and get a plan!
    plan = run_fd(domain_filename, instance_filename, verbose=False)
    if plan is None:
        # instance is unsolvable
        print("Unsolvable instance")
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
    # Note that we don't need to add the last state, as it's been already added as a child of its parent

    if not model.is_goal(s):
        raise RuntimeError(f"State after executing FD plan is not a goal: {s}")

    return allstates


def generate_plan_and_create_sample(domain_filename, instance_filename, model, init, instance_data, sample, initial_sample=False):
    if instance_filename is None:
        # If there is no associated instance filename, we print the given initial state into a PDDL instance file
        assert init is not None
        prob_init = copy.copy(model.problem)
        prob_init.init = init
        instance_filename = generate_instance_file(prob_init, instance_data['pddl_constants'])

    allstates = compute_plan(model, domain_filename, instance_filename, init)

    if allstates is not None:
        # Add the states (plus their expansions) to the D2L Transitionsample object.
        for i, state in enumerate(allstates, start=0):
            # ToDo maybe the initial_sample is unnecessary
            #sid = sample.get_state_id(state)
            #if sid is None or sample.is_expanded(sid):
            #    continue
            expand_state_into_sample(sample, state, instance_data['id'], model, root=((i == 0) and initial_sample))


def expand_state_into_sample(sample, state, instance_id, model, root=False):
    sid = sample.add_state(state, instance_id=instance_id, expanded=True, goal=model.is_goal(state),
                           unsolvable=False, root=root)

    for op, sprime in model.successors(state):
        # Note that we set update_if_duplicate=False to prevent this from updating a previously seen
        # state that is in the plan trace.
        succid = sample.add_state(sprime, instance_id=instance_id, expanded=False, goal=model.is_goal(sprime),
                                  unsolvable=False, root=False, update_if_duplicate=False)
        sample.add_transition(sid, succid)


def random_walk(model, length):
    s = model.problem.init
    for i in range(0, length):
        s = random.choice([succ for _, succ in model.successors(s)])
    return s


def generate_initial_sample(config, all_instance_data):
    # To generate the initial sample, we compute one plan per training instance, and include in the sample all
    # states that are part of the plan, plus all their (possibly unexpanded) children.
    # _run_brfs(config, None, None)

    sample = TransitionSample()
    for instance in config.instances:
        instance_data = all_instance_data[instance]
        model = instance_data['search_model']
        generate_plan_and_create_sample(config.domain, instance, model, model.init(),
                                        instance_data, sample, True)

        for rwi in range(0, config.num_random_walks):
            s = random_walk(model, length=config.random_walk_length)
            generate_plan_and_create_sample(config.domain, None, model, s, instance_data, sample, True)

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

    def search(self, root, verbose=True):
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
                if verbose:
                    logging.info(f"Goal found after {stats.nexpansions} expansions")
                return True, {current.state}

            child, operator = self.policy(current.state)
            if operator is None:
                if verbose:
                    logging.error(f"Policy not defined on state {current.state}")
                return False, {current.state}

            current = make_child_node(current, operator, child)
            if current.state in closed:
                loop = retrieve_loop_states(current)
                if verbose:
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


def test_policy_and_compute_flaws(policy, all_instance_data, instances, config, sample=None):
    """
    If sample is not None, the computed flaws are added to it.
    """
    if sample is not None:
        logging.info(f"Computing policy flaws over {len(instances)} instances")
    nsolved = 0
    for val_instance in instances:
        val_data = all_instance_data[val_instance]

        search_model = val_data['search_model']
        dl_model_factory = val_data['dl_model_factory']
        static_atoms = val_data['static_atoms']
        problem = val_data['problem']
        
        d2l_policy = D2LPolicy(search_model, policy, dl_model_factory, static_atoms)
        search = SafePolicyGuidedSearch(search_model, d2l_policy)

        # Collect all the states from which we want to test the policy
        roots = {problem.init}
        if config.refine_policy_from_entire_sample and sample is not None:
            roots.update(sample.get_leaves(val_data['id']))

        flaws = set()
        for root in roots:
            res, f = search.search(root, verbose=False)
            if not res:
                flaws.update(f)
            if len(flaws) > config.refinement_batch_size:
                break

        if not flaws:
            nsolved += 1
            continue

        # If a sample was provided, add the found flaws to it
        if sample is not None:
            logging.info(f"Adding {len(flaws)} flaws to size-{len(sample.states)} sample")
            for state in flaws:

                if config.compute_plan_on_flaws:
                    generate_plan_and_create_sample(config.domain, None, search_model, state, val_data, sample)
                else:
                    sid = sample.get_state_id(state)
                    if sid is None or not sample.is_expanded(sid):
                        # If the state is already in the sample and already expanded, no need to do anything about it
                        expand_state_into_sample(sample, state, val_data['id'], search_model, root=False)

    return nsolved


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

    config.validation_instances = [os.path.join(BENCHMARK_DIR, config.domain_dir, i) for i in
                                   config.validation_instances]

    all_instances = list(sorted(set(config.validation_instances).union(config.instances)))
    all_instance_data, language = compute_instance_data(all_instances, config)

    # Compute a plan for each of the training instances, and put all states in the plan into the sample, along with
    # all of their (possibly non-expanded) children.
    # All states will need to be labeled with their status (goal / unsolvable / alive)
    sample = generate_initial_sample(config, all_instance_data)
    iteration = 1
    while True:
        # TODO: This could be optimized to avoid recomputing the features over the states that already were in the
        #       sample on the previous iteration.
        logging.info(f"### MAIN ITERATION: {iteration}; SAMPLE SIZE: {sample.num_states()}")
        code, res = generate_feature_pool(config, sample, all_instance_data)
        assert code == ExitCode.Success

        policy = compute_policy_for_sample(config, language)
        if not policy:
            logging.info("No policy found under given complexity bound")
            break

        # Test the policy on the validation set
        nsolved = test_policy_and_compute_flaws(policy, all_instance_data, config.validation_instances, config, sample)
        if nsolved == len(config.validation_instances):
            logging.info("Policy solves all states in training set")
            break  # Policy test was successful, we're done.

        mark_optimal_transitions(sample)

        # Since we have only generated some (optimal) plan, we don't know the actual V* value for states that are in the
        # sample but not in the plan. We mark that accordingly with the special -2 "unknown" value
        for s in sample.states.keys():
            if s not in sample.expanded and s not in sample.goals:
                sample.vstar[s] = -2

        log_sampled_states(sample, config.resampled_states_filename)
        print_transition_matrix(sample, config.transitions_info_filename)

        iteration += 1

    # Run on the test set and report coverage. No need to add flaws to sample here
    if policy:
        all_instances = list(sorted(set(config.test_policy_instances)))
        test_instance_data, _ = compute_instance_data(all_instances, config)
        nsolved = test_policy_and_compute_flaws(policy, test_instance_data, config.test_policy_instances, config)
        print(f"Learnt policy solves {nsolved}/{len(config.test_policy_instances)} in the test set")
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

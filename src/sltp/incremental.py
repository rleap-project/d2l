import copy
import logging
import os
import sys
import tempfile

import numpy as np

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
from .separation import generate_user_provided_policy
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


def run_fd(config, domain, instance, verbose):
    """ Run Fast Downward on the given problem, and return a plan, or None if
    the problem is not solvable. """
    # e.g. fast-downward.py --alias seq-opt-lmcut /home/frances/projects/code/downward-benchmarks/gripper/prob01.pddl

    exp_dir = config.__dict__["experiment_dir"]
    with tempfile.NamedTemporaryFile(mode='w', delete=True) as tf:
        args = f'--alias seq-opt-lmcut --plan-file {exp_dir}/plan.txt {domain} {instance}'.split()
        stdout = None if verbose else tf.name
        retcode = execute(['fast-downward.py'] + args, stdout=stdout)
    if retcode != 0:
        logging.error("Fast Downward error")
        return None

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

    if not model.is_goal(s):
        raise RuntimeError(f"State after executing FD plan is not a goal: {s}")

    allstates.append(s)
    return allstates


def generate_plan_and_create_sample(config, domain_filename, instance_filename, model, flaw, instance_data, sample, initial_sample=False, search=None):
    if instance_filename is None:
        # If there is no associated instance filename, we print the given initial state into a PDDL instance file
        assert flaw is not None
        prob_init = copy.copy(model.problem)
        prob_init.init = flaw['state']
        instance_filename = generate_instance_file(prob_init, instance_data['pddl_constants'])

    allstates = compute_plan(config, model, domain_filename, instance_filename, flaw['state'])

    if allstates is None :
        # Search for the first alive-to-unsolvable transition
        alive, unsolvable = search.get_alive_to_unsolvable(config, flaw['root'], model, domain_filename, instance_data)
        assert unsolvable is not None
        if alive is not None and alive['state'] is not None and alive['vstar'] is not None:
            expand_state_into_sample(sample, alive['state'], instance_data['id'], model, unsolvable=False, vstar=alive['vstar'], root=True)
        expand_state_into_sample(sample, unsolvable['state'], instance_data['id'], model, unsolvable=True, vstar=-1, root=False)

    elif len(allstates) == 1 :
        assert model.is_goal(allstates[0])
    elif initial_sample :
        # Expand all the states besides the goal state
        for i in range(0,len(allstates)-1):
        #for i, state in enumerate(allstates, start=0):
            expand_state_into_sample(sample, allstates[i], instance_data['id'], model, unsolvable=False,
                                     vstar=len(allstates)-1-i, root=(i == 0))
    else :
        # Expand the latest new state of the plan (skip the goal and initial state) and add its optimal transition
        for i in range(len(allstates)-2, 0, -1):
            sid = sample.get_state_id(allstates[i])
            if sid is None or not sample.is_expanded(sid):
                expand_state_into_sample(sample, allstates[i], instance_data['id'], model, unsolvable=False,
                                         vstar=(len(allstates)-1-i), root=False)
                sample.add_optimal_transition(sample.get_state_id(allstates[i]), sample.get_state_id(allstates[i+1]))

        # Expand the first state of the plan
        expand_state_into_sample(sample, allstates[0], instance_data['id'], model, unsolvable=False,
                                 vstar=len(allstates) - 1, root=True)
        # Add the optimal transition between the first and second state of the trace
        sample.add_optimal_transition(sample.get_state_id(allstates[0]), sample.get_state_id(allstates[1]))


def expand_state_into_sample(sample, state, instance_id, model, unsolvable=False, vstar=-2, root=False):
    sid = sample.add_state(state, instance_id=instance_id, expanded=(not unsolvable), goal=model.is_goal(state),
                           unsolvable=unsolvable, vstar=vstar, root=root)

    if not unsolvable:
         for op, sprime in model.successors(state):
             # Note that we set update_if_duplicate=False to prevent this from updating a previously seen
             # state that is in the plan trace.
             succid = sample.add_state(sprime, instance_id=instance_id, expanded=False, goal=model.is_goal(sprime),
                                  unsolvable=False, vstar=-2, root=False, update_if_duplicate=False)
             sample.add_transition(sid, succid)


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

    def add_state(self, state, hstar):
        sid = self.state_to_id.get(state)
        if sid is not None:
            return sid
        sid = self.state_to_id[state] = len(self.states)
        self.states.append(state)
        self.state_to_hstar[state] = hstar
        return sid

    def in_sample(self, state):
        return state in self.state_to_id

    def get_id(self, state):
        return self.state_to_id[state]


class SampleGenerator:
    def __init__(self, config, rng, training_instances_data):
        self.rng = rng
        self.config = config
        self.domain_filename = config.domain
        self.training_instances_data = training_instances_data
        self.sample = SimpleSample()
        self.positive = set()
        self.negative = set()
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
            return sys.maxsize

        sids = []
        for d, state in enumerate(reversed(plan_states), start=0):
            sids.append(self.sample.add_state(state, d))
            if d > 0:
                # By definition, all transitions in an optimal plan are h*-decreasing
                self.positive.add((sids[-1], sids[-2]))
        return len(plan_states)-1

    def compute_hstar(self, model, instance_data, state):
        hstar = self.hstar_cache.get(state)
        if hstar is not None:
            return hstar

        hstar = self.hstar_cache[state] = self.compute_hstar_uncached(model, instance_data, state)
        return hstar

    def add_point(self, sid, succ_id, positive):
        if positive:
            self.positive.add((sid, succ_id))
        else:
            self.negative.add((sid, succ_id))

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
        log_sampled_states(self.sample, self.config.resampled_states_filename)
    
        return self.sample

    def process(self, instance_data, model, s):
        succs_hstar = [(self.compute_hstar(model, instance_data, sprime), sprime) for op, sprime in model.successors(s)]
        hstar = min(h for h, _ in succs_hstar) + 1 if succs_hstar else sys.maxsize
        self.sample.add_state(s, hstar)
        for hsucc, succ in succs_hstar:
            self.add_point(self.sample.get_id(s), self.sample.get_id(succ), hsucc < hstar)

    def process_state(self, instance_data, model, state):

        allstates = self.compute_plan(self.config, model, instance_data['pddl_constants'], state)
        if allstates is None:
            logging.warning(f"Found unsolvable state: {state}")
            return

        # Add the states (plus their expansions) to the D2L Transitionsample object.
        sids = []
        for i, state in enumerate(allstates, start=0):
            hstar = len(allstates) - i
            sids.append(self.sample.add_state(state, instance_data['id'], expanded=True, goal=model.is_goal(state),
                                         unsolvable=False, vstar=hstar, root=(i == 0)))
            if i > 0:
                self.positive.add((sids[-2], sids[-1]))
    
            if i < len(allstates) - 1:
                for op, sprime in model.successors(state):
                    if sprime == allstates[i + 1]:
                        continue
    
                    succid = self.sample.add_state(sprime, instance_id=instance_data['id'], expanded=False,
                                              goal=model.is_goal(sprime), unsolvable=None, vstar=-2, root=False,
                                              update_if_duplicate=False)
                    self.sample.add_transition(sids[-1], succid)
    
                    piprime = self.compute_plan(self.config, model, instance_data['pddl_constants'], sprime)
                    hstar_prime = 9999999999 if piprime is None else len(piprime) - 1
                    if hstar_prime < hstar:
                        self.positive.add((sids[-1], succid))
                    else:
                        self.negative.add((sids[-1], succid))


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
                # print(current.state)
                loop = retrieve_loop_states(current)
                if verbose:
                    logging.error(f"Size-{len(loop)} loop detected after {len(closed)} expansions. State: {current.state}")
                return False, loop

            closed.add(current.state)
            stats.nexpansions += 1

    # ToDo it can be improved with a binary search over the trace
    # reducing the number of calls to the planner
    def get_alive_to_unsolvable(self, config, root, model, domain_filename, instance_data):
        """Run the policy-guided search.
        Return an alive to unsolvable transition from the policy trace.
        It is known that such a transition exists in advanced.
        """
        current = make_root_node(root)
        parent = {'state' : None, 'vstar' : None }

        while True:
            current_data = {'state' : current.state, 'vstar' : -1}

            # Call FD from current.state
            prob_init = copy.copy(model.problem)
            prob_init.init = current.state
            instance_filename = generate_instance_file(prob_init, instance_data['pddl_constants'])
            allstates = compute_plan( config, model, domain_filename, instance_filename, current.state)

            # Alive to unsolvable transition has been found!
            # Parent is None when the root is unsolvable and a leaf of the
            # expanded graph, so the alive state was previously expanded
            if allstates is None :
                return parent, current_data

            current_data['vstar'] = len(allstates) - 1

            child, operator = self.policy(current.state)
            assert operator is not None
            current = make_child_node(current, operator, child)

            parent = current_data




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


def test_policy_and_compute_flaws(policy, all_instance_data, instances, config, sample=None, verbose=False):
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
        roots = [problem.init]
        if config.refine_policy_from_entire_sample and sample is not None:
            roots = roots + list(sample.get_leaves(val_data['id']))

        flaws = []
        for i, root in enumerate(roots):
            res, f = search.search(root, verbose=verbose)
            if not res:
                # Retrieve only the leaf node from a loopy trajectory
                if len(f) > 1:
                    for s in f:
                        sid = sample.get_state_id(s)
                        if sid is not None and not sample.is_expanded(sid):
                            flaw = {'state': s, 'root': root}
                            flaws.append(flaw)
                            break
                # Add the incompatible state to the sample
                else :
                    flaw = {'state': next(iter(f)), 'root': root}
                    flaws.append(flaw)

            if len(flaws) >= config.refinement_batch_size:
                break

        if not flaws:
            nsolved += 1
            continue

        # If a sample was provided, add the found flaws to it
        if sample is not None:
            logging.info(f"Adding {len(flaws)} flaws to size-{len(sample.states)} sample")
            for f in flaws:
                if config.compute_plan_on_flaws:
                    generate_plan_and_create_sample(config, config.domain, None, search_model, f, val_data, sample, False, search)
                else:
                    sid = sample.get_state_id(f['state'])
                    if sid is None or not sample.is_expanded(sid):
                        # If the state is already in the sample and already expanded, no need to do anything about it
                        expand_state_into_sample(sample, f['state'], val_data['id'], search_model, root=False)

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

    #config.validation_instances = [os.path.join(BENCHMARK_DIR, config.domain_dir, i) for i in
    #                               config.validation_instances]

    rng = np.random.default_rng(config.seed)

    # Check if user has provided some policy to be tested
    user_policy = None if config.d2l_policy is None else generate_user_provided_policy(config)

    all_instance_data, language = compute_instance_data(config.instances, config)

    # Compute a plan for each of the training instances, and put all states in the plan into the sample, along with
    # all of their (possibly non-expanded) children.
    # All states will need to be labeled with their status (goal / unsolvable / alive)
    generator = SampleGenerator(config, rng, all_instance_data)
    sample = generator.generate()

    iteration = 1
    while True:
        # TODO: This could be optimized to avoid recomputing the features over the states that already were in the
        #       sample on the previous iteration.
        logging.info(f"### MAIN ITERATION: {iteration}; SAMPLE SIZE: {sample.num_states()}")
        code, res = generate_feature_pool(config, sample, all_instance_data)
        assert code == ExitCode.Success

        if user_policy:
            policy = user_policy
            logging.info("Testing user-provided policy:")
            policy.minimize()
            policy.print_aaai20()
            nsolved = test_policy_and_compute_flaws(user_policy, all_instance_data, config.instances, config, sample, verbose=False)
            logging.info(f"User-provided policy solves {nsolved}/{len(config.instances)} in the training set")
            break

        policy = compute_policy_for_sample(config, language)
        if not policy:
            logging.info("No policy found under given complexity bound")
            break

        # Validate the policy on roots of the training set
        nsolved = test_policy_and_compute_flaws(policy, all_instance_data, config.instances, config, sample)
        if nsolved == len(config.instances):
            logging.info("Policy solves all states in training set")
            break  # Policy test was successful, we're done.

        #mark_optimal_transitions(sample)

        # Since we have only generated some (optimal) plan, we don't know the actual V* value for states that are in the
        # sample but not in the plan. We mark that accordingly with the special -2 "unknown" value
        #for s in sample.states.keys():
        #    if s not in sample.expanded and s not in sample.goals:
        #        sample.vstar[s] = -2

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

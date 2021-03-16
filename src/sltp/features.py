import logging
from typing import Union

from tarski.grounding.ops import approximate_symbol_fluency

from .models import DLModelFactory, FeatureModel
from .util.misc import compute_universe_from_pddl_model, state_as_atoms, types_as_atoms
from tarski.syntax.transform.errors import TransformationError
from tarski.syntax.transform.simplifications import transform_to_ground_atoms

from .language import parse_pddl
from tarski.dl import compute_dl_vocabulary, Feature


def generate_model_cache(domain, instances, sample, parameter_generator):
    parsed_problems = parse_all_instances(domain, instances)
    language, nominals, model_cache, infos, goal_predicates =\
        compute_models(domain, sample, parsed_problems, parameter_generator)
    return language, model_cache


def parse_all_instances(domain, instances):
    logging.info(f"Parsing {len(instances)} PDDL instances")
    return [parse_pddl(domain, instance) for instance in instances]


def compute_static_atoms(problem):
    """ Compute a list with all of the atoms and predicates from the problem that are static """
    init_atoms = state_as_atoms(problem.init)

    fluent_symbols, _ = approximate_symbol_fluency(problem)
    fluent_symbols = set(s.name for s in fluent_symbols)

    static_atoms = set()
    static_predicates = set()
    for atom in init_atoms:
        predicate_name = atom[0]
        if predicate_name == 'total-cost':
            continue

        if predicate_name not in fluent_symbols:
            static_atoms.add(atom)
            static_predicates.add(predicate_name)

    static_predicates.add('object')  # The object predicate is always static
    for atom in types_as_atoms(problem.language):
        predicate_name = atom[0]
        static_atoms.add(atom)
        static_predicates.add(predicate_name)

    return static_atoms, static_predicates


def compute_instance_information(problem, goal_predicates):
    goal_denotations = None

    # Compute the universe of each instance
    universe = compute_universe_from_pddl_model(problem.language)

    static_atoms, static_predicates = compute_static_atoms(problem)

    if goal_predicates:
        goal_denotations = {}  # Atoms indexed by their predicate name
        for p in goal_predicates:
            goal_denotations[p] = list()

        try:
            ground_atoms = transform_to_ground_atoms(problem.goal)
        except TransformationError:
            logging.error("Cannot create goal concepts when problem goal is not a conjunction of ground atoms")
            raise

        for atom in ground_atoms:
            predicate_name = atom[0]
            goal_denotations[predicate_name].append(atom)

    return InstanceInformation(universe, static_atoms, static_predicates, goal_denotations, goal_predicates)


def compute_predicates_appearing_in_goal(problem, use_goal_denotation):
    if not use_goal_denotation:
        return set()

    try:
        ground_atoms = transform_to_ground_atoms(problem.goal)
    except TransformationError:
        logging.error("Cannot create goal concepts when problem goal is not a conjunction of ground atoms")
        raise

    return {atom[0] for atom in ground_atoms}


class InstanceInformation:
    """ A simple collection of instance data necessary to create the DL models """
    def __init__(self, universe, static_atoms, static_predicates, goal_denotations, goal_predicates):
        self.static_atoms = static_atoms
        self.static_predicates = static_predicates
        self.goal_denotations = goal_denotations
        self.goal_predicates = goal_predicates
        self.universe = universe


def report_use_goal_denotation(parameter_generator):
    if parameter_generator is not None:
        logging.info('Using user-provided domain parameters and ignoring goal representation')
        return False
    else:
        logging.info('Using goal representation and no domain parameters')
        return True


def compute_models(domain, sample, parsed_problems, parameter_generator):
    problems = [problem for problem, _, _ in parsed_problems]
    use_goal_denotation = report_use_goal_denotation(parameter_generator)

    goal_predicates = set().union(*(compute_predicates_appearing_in_goal(p, use_goal_denotation) for p in problems))

    infos = [compute_instance_information(problem, goal_predicates) for problem in problems]

    # We assume all problems languages are the same and simply pick the first one
    language = problems[0].language
    vocabulary = compute_dl_vocabulary(language)

    nominals, model_cache = create_model_cache_from_samples(vocabulary, sample, domain, parameter_generator, infos)
    return language, nominals, model_cache, infos, goal_predicates


def compute_nominals(domain, parameter_generator):
    # A first parse without all constants to get exactly those constants that we want as nominals
    _, language, nominals = parse_pddl(domain)
    if parameter_generator is not None:
        nominals += parameter_generator(language)
    return nominals


def create_model_factory(domain, instance, parameter_generator):
    nominals = compute_nominals(domain, parameter_generator)

    problem, language, _ = parse_pddl(domain, instance)
    vocabulary = compute_dl_vocabulary(language)
    use_goal_denotation = report_use_goal_denotation(parameter_generator)

    goal_predicates = compute_predicates_appearing_in_goal(problem, use_goal_denotation)
    info = compute_instance_information(problem, goal_predicates)

    return problem, DLModelFactory(vocabulary, nominals, info)


def create_model_cache_from_samples(vocabulary, sample, domain, parameter_generator, infos):
    """  """
    nominals = compute_nominals(domain, parameter_generator)
    model_cache = create_model_cache(vocabulary, sample.states, sample.instance, nominals, infos)
    return nominals, model_cache


def create_model_cache(vocabulary, states, state_instances, nominals, infos):
    """ Create a DLModelCache from the given parameters"""
    # First create the model factory corresponding to each instance
    model_factories = []
    for info in infos:
        model_factories.append(DLModelFactory(vocabulary, nominals, info))

    # Then create the model corresponding to each state in the sample
    models = {}
    for sid, state in states.items():
        instance = state_instances[sid]
        models[sid] = model_factories[instance].create_model(state)
    return DLModelCache(models)


class DLModelCache:
    def __init__(self, models):
        """ Create a DLModelCache from a dictionary of precomputed models, indexed by corresponding state """
        self.models = models

    def get_term_model(self, state):
        return self.models[state]

    def get_feature_model(self, state):
        return FeatureModel(self.models[state])


def translate_atom(atom):
    assert atom[0] == '(' and atom[-1] == ')'
    return atom[1:-1].split(' ')


def translate_state(state, static_atoms):
    """ Translate a pyperplan-like state into a list with the format required by SLTP's concept denotation processor """
    return [translate_atom(atom) for atom in state] + list(static_atoms)


def generate_model_from_state(model_factory, state, static_atoms):
    translated = translate_state(state, static_atoms)
    return FeatureModel(model_factory.create_model(translated))


class FeatureInterpreter:
    def __init__(self, domain_filename, instance_filename, parameter_generator=None):
        """ Construct a feature interpreter able to interpret D2L features in any state
        of the given instance and domain.

        The optional `parameter_generator` parameter can be a function mapping Tarski languages
        to  lists of Tarski constants representing the parameters of the generalized problem.
        See e.g. the method "blocksworld_parameters_for_on" in the experiments directory.
        Leave this parameter to None if your features contain no parameters.
        """
        self.problem, self.model_factory = create_model_factory(domain_filename, instance_filename, parameter_generator)
        self.static_atoms, _ = compute_static_atoms(self.problem)

    def interpret(self, state: str, feature: Union[str, Feature]):
        """ Return the denotation of the given feature in the given state.
        Both entities can be represented as strings.
        """
        from .util.serialization import unserialize_feature
        model = generate_model_from_state(self.model_factory, state, self.static_atoms)
        if isinstance(feature, str):
            feature = unserialize_feature(self.problem.language, feature)

        return model.denotation(feature)

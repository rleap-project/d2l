"""
Concept and feature models and related classes
"""
from bitarray import bitarray
from tarski.model import Model

from .extensions import uncompress_extension, compress_extension
from .util.misc import try_number
from tarski import Predicate, Function
from tarski.dl import UniversalConcept, EmptyConcept, NullaryAtom, PrimitiveConcept, PrimitiveRole, GoalRole, \
    GoalConcept, GoalNullaryAtom, NominalConcept, EmpiricalBinaryConcept, NullaryAtomFeature
from tarski.syntax import Sort

_STANDARD_DL_MAPPING = {0: NullaryAtom, 1: PrimitiveConcept, 2: PrimitiveRole}
_GOAL_DL_MAPPING = {0: GoalNullaryAtom, 1: GoalConcept, 2: GoalRole}
_TOP = UniversalConcept('object')
_BOT = EmptyConcept('object')


class DLModel:
    """ A Description Logic model, able to return the denotation of any concept or role (or nullary atom) in the state
    that represents this model (each state implicitly represents a FOL model).

    The following elements are kept as a link to a single "static" dictionary, since their denotation will surely not
    change over different states (of the same instance):
      - The universe denotation
      - The denotation of primitive predicates
      - The denotation of goal concepts and roles
      - The denotation of nominal concepts
      - Compound concepts and roles involving only the above elements [TODO: STILL NEEDS TO BE IMPLEMENTED]
    """
    def __init__(self, primitive_denotations, statics, universe_idx, cache=None):
        self.primitive_denotations = primitive_denotations
        self.statics = statics
        self.cache = cache
        self.universe_idx = universe_idx
        self._universe = self.primitive_denotation(_TOP)
        self.dencache = dict()

    def universe(self):
        return self._universe

    def denotation(self, term):
        try:
            return self.dencache[term]
        except KeyError:
            x = self.dencache[term] = term.denotation(self)
            return x

    def primitive_denotation(self, term):
        if term in self.statics:
            return self.statics[term]
        return self.primitive_denotations[term]

    def compressed(self, data, arity):
        if isinstance(data, bitarray):
            return data
        return compress_extension(data, len(self._universe), arity)

    def uncompressed(self, data, arity):
        if not isinstance(data, bitarray):
            return data
        return uncompress_extension(data, len(self._universe), arity)

    def uncompressed_denotation(self, term):
        return self.uncompressed(self.denotation(term), term.ARITY)

    def compressed_denotation(self, term):
        return self.compressed(self.denotation(term), term.ARITY)


def default_denotation_by_arity(arity):  # The default values for each possible arity
    if arity == 0:
        return False
    return set()


def tarski_atom_to_tuple(atom):
    return (atom.predicate.name, ) + tuple(st.name for st in atom.subterms)


class DLModelFactory:
    """ A factory of DL models tailored to a concrete universe of discourse.
    This means that the factory is suitable to generate models that work for *a single planning instance*,
    unless all of your instances happen to have the same universe. Even in that case, in the future we might want to
    implement some precomputations based on the goal and static information of the instance.
    """
    def __init__(self, vocabulary, nominals, instance_info):
        """ `vocabulary` should contain a mapping from (relevant) symbol names to the actual Tarski symbol object """
        self.instance_info = instance_info
        self.universe = instance_info.universe
        self.vocabulary = vocabulary  # vocabulary contains pred./function symbols relevant for any interpretation
        self.base_denotations, self.statics = self.compute_base_denotations(vocabulary, nominals, instance_info)

    def compute_base_denotations(self, vocabulary, nominals, instance_info):
        """ Initialize the data structure that we will use to store the denotation that corresponds
            to each logical symbol in the language. For instance, the denotation of a unary symbol such as "clear"
            will be represented as a set of objects; here we just create a mapping from the "clear" symbol to
            an empty set. This incidentally enforces the closed-world assumption:
            predicates not appearing on the state trace will be assumed to have empty denotation"""
        denotations = {}
        statics = {_TOP: instance_info.universe.as_extension(), _BOT: set()}
        for p in vocabulary.values():
            arity = 1 if isinstance(p, Sort) else p.uniform_arity()
            name = p.name
            dl_element = _STANDARD_DL_MAPPING[arity](p)
            if name in instance_info.static_predicates:
                statics[dl_element] = default_denotation_by_arity(arity)
            else:
                denotations[dl_element] = default_denotation_by_arity(arity)

            if instance_info.goal_denotations is not None and name in instance_info.goal_denotations:
                dl_element = _GOAL_DL_MAPPING[arity](p)
                statics[dl_element] = default_denotation_by_arity(arity)

        #
        for nominal in nominals:
            statics[NominalConcept(nominal.symbol, nominal.sort)] = {self.universe.index(nominal.symbol)}

        # If a goal was passed, add the goal denotations (e.g. clear_G) computed from the goal "partial state"
        if instance_info.goal_denotations is not None:
            for dens in instance_info.goal_denotations.values():
                for atom in dens:
                    self.process_atom(atom, statics, _GOAL_DL_MAPPING)

        # Add the denotation of all static atoms
        for atom in instance_info.static_atoms:
            self.process_atom(atom, statics, _STANDARD_DL_MAPPING)

        return denotations, statics

    def base_model(self):
        def process(k, v):  # A cheap copy of the dictionary of denotations
            if isinstance(v, bool):
                return v
            return v.copy()
        return {k: process(k, v) for k, v in self.base_denotations.items()}

    def create_model(self, state):
        """ Create a model capable of interpreting any DL concept / role under the given state """
        assert isinstance(state, Model)
        # Start with a copy of all the precomputed data
        denotations = self.base_model()
        _ = [self.process_atom(tarski_atom_to_tuple(atom), denotations, _STANDARD_DL_MAPPING)
             for atom in state.as_atoms()
             if atom.predicate.name not in self.instance_info.static_predicates]
        return DLModel(denotations, self.statics, self.universe)

    def process_atom(self, atom, denotations, dl_mapping):
        """ Process an atom represented in format e.g. ("on", "a", "b") and add the corresponding modifications
        to the `denotations` dictionary.
        """
        assert len(atom) <= 3, "Cannot deal with arity>2 predicates or arity>1 functions yet"
        symbol = self.vocabulary[atom[0]]
        assert isinstance(symbol, (Predicate, Function, Sort))
        arity = 1 if isinstance(symbol, Sort) else symbol.uniform_arity()
        assert arity == len(atom) - 1
        dl_element = dl_mapping[arity](symbol)

        if len(atom) == 1:  # i.e. a nullary predicate
            denotations[dl_element] = True

        elif len(atom) == 2:  # i.e. a unary predicate or nullary function
            denotations[dl_element].add(self.universe.index(try_number(atom[1])))

        else:  # i.e. a binary predicate or unary function
            t = (self.universe.index(try_number(atom[1])), self.universe.index(try_number(atom[2])))
            denotations[dl_element].add(t)


class FeatureModel:
    """ """
    def __init__(self, concept_model):
        assert isinstance(concept_model, DLModel)
        self.concept_model = concept_model

    def denotation(self, feature):
        val = feature.denotation(self.concept_model)
        return bool(val) if isinstance(feature, (EmpiricalBinaryConcept, NullaryAtomFeature)) else val

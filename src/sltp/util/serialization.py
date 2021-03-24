import logging
import pickle

from tarski import Predicate, Function
from tarski.dl import NullaryAtom, EmpiricalBinaryConcept, ConceptCardinalityFeature, PrimitiveConcept, \
    UniversalConcept, NotConcept, ExistsConcept, ForallConcept, PrimitiveRole, EmptyConcept, AndConcept, GoalRole, \
    GoalConcept, InverseRole, EqualConcept, StarRole, NullaryAtomFeature, NominalConcept, MinDistanceFeature, \
    RestrictRole, OrConcept  # , ConditionalFeature, RoleDifference
from tarski.dl.features import DifferenceFeature
from tarski.syntax import Sort


def serialize(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # f.write(pickle.dumps(data))
        # f.write(jsonpickle.encode(data, keys=True))


def deserialize(filename):
    with open(filename, 'rb') as f:
        try:
            data = pickle.load(f)
            # data = jsonpickle.decode(f.read(), keys=True)
        except EOFError as e:
            logging.error("Deserialization error: couldn't unpicle file '{}'".format(filename))
            raise
    return data


def unserialize_features(language, filename, indexes=None):
    features = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file, 0):
            if indexes is None or i in indexes:
                name, complexity = line.rstrip('\n').split("\t")
                features.append(unserialize_feature(language, name, int(complexity)))
    return features


def unserialize_feature(language, string, complexity=None):
    """ Construct an SLTP feature from the given serialized string representation
    If given, set the complexity of the feature to the given value.
    """
    if string.startswith("If{"):  # special treatment for recursive If features
        assert False
        assert string.endswith("}{Infty}")  # So far we only handle conditional features of else-infty type
        concept = string[3:-8]
        condition, body = [unserialize_feature(language, s) for s in concept.split('}{')]
        return ConditionalFeature(condition, body)

    if string.startswith("LessThan{"):
        content = string[len("LessThan{"):-1]
        f1, f2 = [unserialize_feature(language, s) for s in content.split('}{')]
        return DifferenceFeature(f1, f2)

    ftype, concept, _ = string.replace("[", "]").split("]")  # Split feature name in 3 parts by '[', ']'
    assert not _
    if ftype == "Atom":
        p = language.get(concept)  # Concept will necessarily be a nullary predicate
        assert p.arity == 0
        return NullaryAtomFeature(NullaryAtom(p))

    elif ftype == "Bool":
        c = parse_concept(language, concept)
        if complexity is not None:
            c.size = complexity
        return EmpiricalBinaryConcept(ConceptCardinalityFeature(c))

    elif ftype == "Num":
        c = parse_concept(language, concept)
        if complexity is not None:
            c.size = complexity
        return ConceptCardinalityFeature(c)

    elif ftype == "Dist":
        parts = concept.split(';')
        assert len(parts) == 3
        params = [parse_concept(language, c) for c in parts]
        return MinDistanceFeature(*params)
    
    raise RuntimeError("Don't know how to unserialize feature \"{}\"".format(string))


parse_feature = unserialize_feature  # Just a handy alias


def parse_concept(language, string):
    ast = read_ast(string)
    # print("{}:\t\t {}".format(string, tree))
    return build_concept(language, ast)


def build_concept(language, node):
    if isinstance(node, str):  # Must be a primitive symbol or universe / empty
        if node == "<universe>":
            return UniversalConcept('object')
        if node == "<empty>":
            return EmptyConcept('object')

        is_goal = False
        if node[-2:] == "_g":
            # We have a goal concept / role
            # (yes, this will fail miserably if some primitive in the original domain is named ending with _g)
            is_goal = True
            node = node[:-2]

        obj = language.get(node)
        assert isinstance(obj, (Predicate, Function, Sort))
        arity = 1 if isinstance(obj, Sort) else obj.uniform_arity()
        assert arity in (1, 2)
        if is_goal:
            return GoalConcept(obj) if arity == 1 else GoalRole(obj)
        else:
            return PrimitiveConcept(obj) if arity == 1 else PrimitiveRole(obj)

    elif node.name == "Nominal":
        assert len(node.children) == 1
        return NominalConcept(node.children[0], language.Object)

    elif node.name == "Not":
        assert len(node.children) == 1
        return NotConcept(build_concept(language, node.children[0]), language.Object)

    elif node.name == "Inverse":
        assert len(node.children) == 1
        return InverseRole(build_concept(language, node.children[0]))

    elif node.name in ("And", "Or"):
        assert len(node.children) == 2
        constructor = AndConcept if node.name == "And" else OrConcept
        return constructor(build_concept(language, node.children[0]), build_concept(language, node.children[1]), 'object')

    elif node.name == "Exists":
        assert len(node.children) == 2
        role = build_concept(language, node.children[0])
        concept = build_concept(language, node.children[1])
        return ExistsConcept(role, concept)

    elif node.name == "Forall":
        assert len(node.children) == 2
        role = build_concept(language, node.children[0])
        concept = build_concept(language, node.children[1])
        return ForallConcept(role, concept)

    elif node.name == "Restrict":
        assert len(node.children) == 2
        role = build_concept(language, node.children[0])
        concept = build_concept(language, node.children[1])
        return RestrictRole(role, concept)

    elif node.name == "Equal":
        assert len(node.children) == 2
        r1 = build_concept(language, node.children[0])
        r2 = build_concept(language, node.children[1])
        return EqualConcept(r1, r2, 'object')

    elif node.name == "Star":
        assert len(node.children) == 1
        r1 = build_concept(language, node.children[0])
        return StarRole(r1)

    elif node.name == "RoleDifference":
        assert False
        assert len(node.children) == 2
        r1 = build_concept(language, node.children[0])
        r2 = build_concept(language, node.children[1])
        return RoleDifference(r1, r2)

    else:
        raise RuntimeError("Don't know how to deserialize concept / feature: {}".format(node))


def read_subitems(string):
    copen = 0
    splits = []
    elems = []
    last_split = -1
    for i, c in enumerate(string):
        if c == "(":
            copen += 1
        elif c == ")":
            copen -= 1
        elif c == ",":
            if copen == 0:
                splits.append(i)
                elems.append(string[last_split+1:i])
                last_split = i
    elems.append(string[last_split+1:])
    return [read_ast(elem) for elem in elems]


def read_ast(string):
    # print("recursive-call on |{}|".format(string))
    lindex = string.find('(')
    rindex = string.rfind(')')

    assert bool(lindex == -1) == bool(rindex == -1)
    if lindex == -1:
        assert rindex == -1
        return string

    assert rindex != -1
    name = string[:lindex]
    between = string[lindex + 1:rindex]
    return Node(name, children=read_subitems(between))


class Node:
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        if not self.children:
            return self.name
        return "{}({})".format(self.name, ', '.join(map(str, self.children)))
    __repr__ = __str__


from ..extensions import UniverseIndex


def update_dict(d, **kwargs):
    res = d.copy()
    res.update(kwargs)
    return res


def try_number(x):
    """ Return the number-like version of x, if x is a number, or x, otherwise"""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


# The commented code below was used for creating a universe index from a sample
# of states, but now we create that from the actual PDDL model instead, which
# should be more reliable
# def compute_universe_from_state_set(states):
#     """ Iterate through all states and collect all possible PDDL objects """
#     universe = UniverseIndex()
#
#     for sid, state in states.items():
#         for atom in state:
#             assert atom
#             if atom[0] == "=":  # Functional atom
#                 assert len(atom) <= 4, "Cannot deal with arity>1 functions yet"
#                 [universe.add(try_number(obj)) for obj in atom[2:]]
#             else:
#                 assert len(atom) in (1, 2, 3), "Cannot deal with arity>2 predicates yet"
#                 [universe.add(try_number(obj)) for obj in atom[1:]]
#
#     universe.finish()  # No more objects possible
#     return universe


def compute_universe_from_pddl_model(language):
    """ Compute a Universe Index from the PDDL model """
    universe = UniverseIndex()
    _ = [universe.add(try_number(c.symbol)) for c in language.constants()]
    universe.finish()  # No more objects possible
    return universe


def state_as_atoms(state):
    """ Transform any state (i.e. interpretation) into a flat set of tuples,
    one per ground atom that is true in the state """
    atoms = set()
    for signature, elems in state.list_all_extensions().items():
        name = signature[0]
        # We unwrap the tuples of Constants into tuples with their (string) names
        atoms.update((name, ) + tuple(o.symbol for o in elem) for elem in elems)
    return atoms


def types_as_atoms(lang):
    """ Generate a list of atoms related to the types of the language, e.g. in spanner:
    man(bob), spanner(spanner1), spanner(spanner2), etc. would be returned as atoms """
    atoms = set()
    for s in lang.sorts:
        # This takes into account type inheritance, as s.domain() contains constants of type s and of derived types.
        if s != lang.Object and s.name != 'number' and not s.builtin:
            atoms.update((s.name, c.symbol) for c in s.domain())
    return atoms


def print_set(data, index):
    return "{{{}}}".format(", ".join(index.value(x) for x in data))


def print_relation(data, index):
    return "{{{}}}".format(", ".join(sorted(print_tuple(x, index) for x in data)))


def print_tuple(tup, index):
    return "({})".format(", ".join(index.value(x) for x in tup))


def extend_namer_to_all_features(namer):
    """ Extend possible feature names to both their numeric and boolean versions"""
    items = list(namer.items())
    atoms = [(k, v) for k, v in items if k.startswith("Atom[")]
    dists = [(k, v) for k, v in items if k.startswith("Dist[")]
    diffs = [(k, v) for k, v in items if k.startswith("LessThan{")]
    return dict(atoms + dists + diffs +
                [("Bool[{}]".format(k), v) for k, v in items] + [("Num[{}]".format(k), v) for k, v in items])


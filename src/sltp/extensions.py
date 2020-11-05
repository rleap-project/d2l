"""
 Code to optimize the handling of extensions (i.e. denotations) of concepts / roles
 in a certain state, and in a trace of states. An "extension trace" is indeed an ordered
 sequence of extensions, containing one extension for every state in a certain trace
 or sample of states. We want extension traces to have a small footprint and to be
 hashable and comparable, so that we can use them in hash tables.
"""
import itertools

from bitarray import bitarray

from tarski.dl import PrimitiveRole, PrimitiveConcept, UniversalConcept, EmptyConcept, NullaryAtom


_btrue = bitarray('1')
_bfalse = bitarray('0')


class ExtensionTrace:
    """
    We represent an extension trace by a single, compact bitmap that contains in sequence the extension for
    each state in the sequence of states from which we build the trace.

    Let `m` be the cardinality of our universe of discourse and `n` the number of states in our state sample.
    Object IDs thus range from 0 to m-1, both inclusive.
    For a concept c, the extension of c in a certain interpretation (i.e. search state) is represented as a bitmap BM
    of length `m`, where BM[i] iff the object with ID `i` belongs to the extension. The extension trace of `c` is simply
    the (ordered) concatenation of the individual extensions into a bitmap of length m*n.

    For a role r, the extension of r in any interpretation is a set of pairs, and is represented as a bitmap BM of
    length `m`^2, where BM[m*i+j], for 0 <= i,j < m, iff the pair (i,j) is in the extension of the role.
    The extension trace of `r` is again the concatenation of the individual extensions into a bitmap that this time
    will have length m^2*n.
    """
    def __init__(self, b_array):
        self.data = b_array
        self.hash = hash(self.data.tobytes())  # We cache the hash for faster retrieval.

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                hasattr(other, 'hash') and self.hash == other.hash and
                self.data == other.data)

    def __str__(self):
        return str(self.data)

    __repr__ = __str__


class UniverseIndex:
    """ An index of the universe :-) Basically mapping every (string) in our universe of discourse to a numeric ID """
    def __init__(self):
        self._objects = []
        self._index = dict()
        self.closed = False

    def index(self, obj):
        assert self.closed
        return self._index[obj]

    def value(self, i):
        assert self.closed
        return self._objects[i]

    def add(self, obj):
        assert not self.closed
        if obj in self._index:
            return
        self._index[obj] = True

    def as_extension(self):
        assert self.closed
        return set(range(0, len(self._objects)))  # i.e. (0, 1, 2, ...)

    def __str__(self):
        ast = '*' if not self.closed else ''
        return "Universe{}({})".format(ast, ', '.join(self._objects))

    __repr__ = __str__

    def finish(self):
        # We sort the objects alphabetically and recreate the dictionary with the new indexes
        self._objects = sorted(self._index.keys(), key=lambda x: str(x))
        self._index = {obj: i for i, obj in enumerate(self._objects, 0)}
        self.closed = True

    def __len__(self):
        return len(self._objects)


def compress_unary_extension(extension, m):
    bits = [False] * m
    for i in extension:
        assert i < m
        bits[i] = True
    return bitarray(bits)


def uncompress_unary_extension(extension, m):
    assert isinstance(extension, bitarray)
    return set(i for i in extension.itersearch(_btrue))


def compress_binary_extension(extension, m):
    assert isinstance(extension, set)
    return compress_unary_extension((m*x+y for x, y in extension), m*m)


def uncompress_binary_extension(extension, m):
    assert isinstance(extension, bitarray)
    return set((i//m, i % m) for i in extension.itersearch(_btrue))


def compress_extension(extension, m, arity):
    assert arity in (1, 2)
    return compress_unary_extension(extension, m) if arity == 1 else compress_binary_extension(extension, m)


def uncompress_extension(extension, m, arity):
    assert arity in (1, 2)
    return uncompress_unary_extension(extension, m) if arity == 1 else uncompress_binary_extension(extension, m)


def trace_to_bits(trace, m, arity):
    """ Transform a trace in format [{a,c,b}, {b,d}, ...] into an (uncompressed) bitmap representation:
        [[1110], [0101], ...] which will be later fed to the bitarray constructor
    """
    assert isinstance(trace, list)
    mapper = compress_unary_extension if arity == 1 else compress_binary_extension
    return itertools.chain.from_iterable(mapper(ext, m) for ext in trace)


# def bits_to_trace(bits, m):
#     as_list = bits.tolist()
#     chunked = (as_list[pos:pos + m] for pos in range(0, len(as_list), m))
#     indexed = [set(i for i, x in enumerate(l) if x is True) for l in chunked]
#     return indexed


def get_term_key(term):
    """ Return the key corresponding to the given DL term.
    For primitives we use a the string representation, otherwise we use the DL term itself.
    The only reason for this is that often when unserializing DL elements from disk in different processes,
    hashes change (as Python salts the hash computations with process-dependent values), so that equality comparisons
    and indexing stop working. This way, primitives are at least recognized and the denotation of any DL
    element can be rebuilt if necessary. It is not an optimal strategy, but works for our current purposes. """
    if isinstance(term, (PrimitiveConcept, PrimitiveRole, UniversalConcept, EmptyConcept, NullaryAtom)):
        return str(term)
    return term

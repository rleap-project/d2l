import itertools
import os
from collections import defaultdict

from natsort import natsorted

from tarski.dl import FeatureValueChange, NullaryAtomFeature, EmpiricalBinaryConcept, ConceptCardinalityFeature

from .language import parse_pddl
from .util.serialization import unserialize_feature
from .util.tools import IdentifiedFeature
from .util.tools import load_selected_features
from .returncodes import ExitCode


def compute_good_transitions(assignment, wsat_varmap_filename):
    """ Reads off the map of good-variables produced by the CNF generator, then takes the WSAT problem solution
    assignment, and computes which are the good transitions. """
    good = []
    with open(wsat_varmap_filename, 'r') as f:
        for line in f:
            var, s, t = map(int, line.rstrip().split())
            if assignment[var] is True:
                good.append((s, t))
    return list(sorted(good))


def read_selected_variable_ids(assignment, filename):
    selected = []
    with open(filename, 'r') as f:
        for line in f:
            featid, varid, name = line.rstrip().split("\t")
            if assignment[int(varid)] is True:
                selected.append(int(featid))
    return selected


def print_maxsat_solution(assignment, filename):
    """ """
    solution = []
    vs = dict()
    with open(filename, 'r') as f:
        for line in f:
            id_, name = line.rstrip().split("\t")
            if assignment[int(id_)] is True:
                # e.g. V(4, 1)
                if name.startswith("V("):
                    state, value = map(int, name[2:-1].split(", "))
                    if state in vs:
                        raise RuntimeError(f"Maxsat solution has more than one value V(s) for state {state}")
                    vs[state] = value
                else:
                    solution.append(name)
    print('\n'.join(natsorted(solution, key=lambda y: y.lower())))
    print('\n'.join(f"V({s}) = {vs[s]}" for s in sorted(vs.keys())))


def generate_policy_from_sat_solution(config, solution, model_cache, print_policy=True):
    assert solution.solved
    features = extract_features_from_sat_solution(config, solution)
    assert features
    # print_maxsat_solution(solution.assignment, config.wsat_allvars_filename)
    good_transitions = compute_good_transitions(solution.assignment, config.wsat_varmap_filename)
    policy = TransitionClassificationPolicy(features)
    for (s, t) in good_transitions:
        m_s = model_cache.get_feature_model(s)
        m_t = model_cache.get_feature_model(t)

        clause = []
        for f in features:
            den_s = f.denotation(m_s)
            den_t = f.denotation(m_t)
            clause += compute_policy_clauses(f, den_s, den_t)

        policy.add_clause(frozenset(clause))

    if print_policy:
        policy.print_aaai20()
    return features, good_transitions, policy


def extract_features_from_sat_solution(config, solution):
    selected_feature_ids = read_selected_variable_ids(
        solution.assignment, os.path.join(config.experiment_dir, "selecteds.wsat"))
    language = config.language_creator(config)
    features = load_selected_features(language, selected_feature_ids, config.serialized_feature_filename)
    features = [IdentifiedFeature(f, i, config.feature_namer(str(f))) for i, f in zip(selected_feature_ids, features)]
    return features


def generate_user_provided_policy(config):
    rules = config.d2l_policy()
    _, language, _ = parse_pddl(config.domain)
    policy = TransitionClassificationPolicy.parse(rules, language, config.feature_namer)
    return policy


def compute_policy_clauses(f, den_s, den_t):
    diff = f.feature.diff(den_s, den_t)

    # if isinstance(f.feature, NullaryAtomFeature) or isinstance(f.feature, EmpiricalBinaryConcept):
    #     if diff in (FeatureValueChange.DEL, FeatureValueChange.ADD):
    #         # For binary features that change their value across a transition, the origin value is implicit
    #         return [DNFAtom(f, diff)]
    #
    # if isinstance(f.feature, ConceptCardinalityFeature):
    #     if diff in (FeatureValueChange.DEC, ):
    #         # Same for numeric features that decrease their value: they necessarily need to start at >0
    #         return [DNFAtom(f, diff)]

    # Else, the start value is non-redundant info that we want to use
    return [DNFAtom(f, den_s != 0), DNFAtom(f, diff)]


def minimize_dnf_policy(dnf):
    while True:
        p1, p2, new = attempt_dnf_merge(dnf)
        if p1 is None:
            break

        # Else do the actual merge
        # newstr = ' AND '.join(sorted(map(str, new)))
        # p1str = ' AND '.join(sorted(map(str, p1)))
        # p2str = ' AND '.join(sorted(map(str, p2)))
        # print(f'Inserting:\n\t"{newstr}"\nRemoving:\n\t"{p1str}"\nRemoving:\n\t"{p2str}"\n')
        dnf.remove(p1)
        dnf.remove(p2)
        dnf.add(new)
    return dnf


def attempt_dnf_merge(dnf):
    for p1, p2 in itertools.combinations(dnf, 2):
        diff = p1.symmetric_difference(p2)
        diffl = list(diff)

        if len(diffl) != 2:  # More than one feature with diff value
            continue

        atom1, atom2 = diffl
        if atom1.feature != atom2.feature:  # Not affecting the same feature
            continue

        if {atom1.value, atom2.value} == {True, False}:
            # The two conjunctions differ in that one has one literal L and the other its negation, the rest being equal
            p_merged = p1.difference(diff)
            return p1, p2, p_merged  # Meaning p1 and p2 should be merged into p_merged

    return None, None, None


class DNFAtom:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def is_state_feature(self):
        return isinstance(self.value, bool)

    def __str__(self):
        if self.is_state_feature():
            return f'{self.feature}>0' if self.value else f'{self.feature}=0'
        # else, we have a transition feature
        return f'{self.feature} {str(self.value).upper()}s'
    __repr__ = __str__

    def __hash__(self):
        return hash((self.feature, self.value))

    def __eq__(self, other):
        return self.feature == other.feature and self.value == other.value


def changes_are_analogous(change1, change2):
    normalization = {
        FeatureValueChange.ADD: FeatureValueChange.INC,
        FeatureValueChange.DEL: FeatureValueChange.DEC,
    }
    return normalization.get(change1, change1) == normalization.get(change2, change2)


class TransitionClassificationPolicy:
    def __init__(self, features):
        self.features = features
        self.dnf = set()

    def add_clause(self, clause):
        self.dnf.add(clause)

    def minimize(self):
        self.dnf = minimize_dnf_policy(self.dnf)

    def transition_is_good(self, m0, m1):
        # If the given transition satisfies any of the clauses in the DNF, we consider it "good"
        return any(self.does_transition_satisfy_clause(clause, m0, m1) for clause in self.dnf)

    def explain_why_transition_is_bad(self, m0, m1):
        for clause in self.dnf:
            print(f'\tClause: {self.print_clause(clause)}')
            self.does_transition_satisfy_clause(clause, m0, m1, explain=True)

    @staticmethod
    def does_transition_satisfy_clause(clause, m0, m1, explain=False):
        for atom in clause:
            feat = atom.feature

            if atom.is_state_feature():
                state_val = feat.denotation(m0) != 0
                if state_val != atom.value:
                    if explain:
                        print(f'\t\t{atom} not satisfied by transition values ({feat.denotation(m0)}, {feat.denotation(m1)})')
                    return False
            else:
                tx_val = feat.feature.diff(feat.denotation(m0), feat.denotation(m1))
                if not changes_are_analogous(tx_val, atom.value):
                    if explain:
                        print(f'\t\t{atom} not satisfied by transition values ({feat.denotation(m0)}, {feat.denotation(m1)})')
                    return False
        return True

    def print(self):
        self.print_header()
        print("Policy:")
        for i, clause in enumerate(self.dnf, start=1):
            print(f"  {i}. " + self.print_clause(clause))

    def print_aaai20(self):
        self.print_header()
        # Group by state features
        grouped = defaultdict(list)
        for clause in self.dnf:
            key = frozenset(atom for atom in clause if atom.value in {True, False})
            value = frozenset(atom for atom in clause if atom.value not in {True, False})
            grouped[key].append(value)

        distinct_values = defaultdict(set)
        for key in grouped.keys():
            for atom in key:
                distinct_values[atom.feature].add(atom.value)

        invariant_value_features = {f: next(iter(vals)) for f, vals in distinct_values.items() if len(vals) == 1}
        print("Invariants: " + ','.join(f"{f}{'>0' if v is True else '=0'}" for f, v in invariant_value_features.items()))
        print("Policy:")
        for i, (statef, transitionf) in enumerate(grouped.items(), start=1):
            state_conds = ' AND '.join(sorted(map(str, statef)))
            feature_conds = ', '.join(self.print_effect_list(e) for e in transitionf)
            print(f"  {i}. {state_conds} -> {feature_conds}")

    @staticmethod
    def print_effect_list(effect):
        return '{' + ', '.join(sorted(map(str, effect))) + '}'

    def print_header(self):
        max_k = max(f.feature.complexity() for f in self.features)
        total_k = sum(f.feature.complexity() for f in self.features)
        print(f"Features (#: {len(self.features)}; total k: {total_k}; max k = {max_k}):")
        for f in self.features:
            print(f"  {f} [k={f.feature.complexity()}]")

    @staticmethod
    def print_clause(clause):
        return ' AND '.join(sorted(map(str, clause)))

    @staticmethod
    def parse(rules, language, feature_namer):
        """ Create a classification policy from a set of strings representing the clauses """
        policy = TransitionClassificationPolicy(features=[])

        allfeatures = dict()

        for clause in rules:
            atoms = []
            for feature_str, value in clause:
                f = allfeatures.get(feature_str)
                if f is None:
                    f = unserialize_feature(language, feature_str)
                    allfeatures[feature_str] = f = IdentifiedFeature(f, len(allfeatures), feature_namer(str(f)))

                # Convert the value to an object
                value = {
                    "=0": False,
                    ">0": True,
                    "INC": FeatureValueChange.INC,
                    "NIL": FeatureValueChange.NIL,
                    "DEC": FeatureValueChange.DEC,
                    "ADD": FeatureValueChange.ADD,
                    "DEL": FeatureValueChange.DEL,
                }[value]

                atoms.append(DNFAtom(f, value))

            policy.add_clause(frozenset(atoms))

        return policy

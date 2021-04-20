import itertools
import logging
import math
from collections import defaultdict, OrderedDict, deque

from .outputs import print_transition_matrix
from .returncodes import ExitCode
from .util.command import read_file
from .util.naming import filename_core


class TransitionSample:
    """ """
    def __init__(self):
        self.states = OrderedDict()
        self.state_to_id = dict()
        self.transitions = defaultdict(set)
        self.parents = defaultdict(set)
        self.roots = set()  # The set of all roots
        self.goals = set()
        self.alive_states = set()
        self.optimal_transitions = set()
        self.expanded = set()
        self.unsolvable = set()
        self.instance = dict()  # A mapping between states and the problem instances they came from
        self.vstar = {}

    def get_state_id(self, state):
        return self.state_to_id.get(state, None)

    def get_state(self, state_id):
        return self.states.get(state_id)

    def is_expanded(self, state_id):
        return state_id in self.expanded

    def is_unsolvable(self, state_id):
        return state_id in self.unsolvable

    def is_goal(self, state_id):
        return state_id in self.goals

    def _register_new_state(self, state):
        sid = len(self.states)
        self.states[sid] = state
        self.state_to_id[state] = sid
        return sid

    def _update_state_info(self, sid, instance_id, expanded, goal, unsolvable, root):
        self.instance[sid] = instance_id
        if expanded:
            self.expanded.add(sid)
        else:
            self.expanded.discard(sid)

        if unsolvable:
            self.unsolvable.add(sid)
        else:
            self.unsolvable.discard(sid)

        if goal:
            self.goals.add(sid)
        else:
            self.goals.discard(sid)

        if root:
            self.roots.add(sid)
        else:
            self.roots.discard(sid)

    def add_state(self, state, instance_id, expanded, goal, unsolvable, root, update_if_duplicate=True):
        """ Add a state and associated info to the sample. """
        sid = self.state_to_id.get(state)
        if sid is not None:  # The state is already in the sample
            if not update_if_duplicate:
                return sid

        else:  # Otherwise, the state is new
            sid = self._register_new_state(state)

        self._update_state_info(sid, instance_id, expanded, goal, unsolvable, root)
        return sid

    def add_transition(self, s, t):
        assert s in self.states and t in self.states
        self.transitions[s].add(t)
        self.parents[t].add(s)

    def get_leaves(self):
        for sid, s in self.states.items():
            if not self.is_expanded(sid) and not self.is_unsolvable(sid) and not self.is_goal(sid):
                yield s

    def add_transitions(self, states, transitions, instance_id, unsolvable):
        """ Add a batch of states coming from the same instance to the sample.
        `states` is expected to be a dictionary mapping state IDs to tuples with all atoms that are true in the
        state (including static atoms).
        `transitions` is expected to be a dictionary mapping state IDs X to sets of IDS Y such that (X, Y) is a transition.
        """
        assert not any(s in self.states for s in states.keys())  # Don't allow repeated states
        self.states.update(states)
        self.transitions.update(transitions)
        self.parents.update(compute_parents(transitions))
        for s in states:
            assert s not in self.instance
            self.instance[s] = instance_id
        self.unsolvable.update(unsolvable)
        # We consider a state expanded if it has some child or it is marked as a deadend
        self.expanded.update(s for s in states if (s in transitions.keys() and len(transitions[s]) > 0) or s in self.unsolvable)

    def mark_as_goals(self, goals):
        self.goals.update(goals)

    def num_states(self):
        return len(self.states)

    def num_transitions(self):
        return sum(len(x) for x in self.transitions.values())

    def mark_as_optimal(self, optimal):
        self.optimal_transitions.update(optimal)

    def mark_as_alive(self, states):
        self.alive_states.update(states)

    def compute_optimal_states(self, include_goals):
        """ Return those states that are the source of an optimal transition """
        states = set(itertools.chain.from_iterable(self.optimal_transitions))
        if include_goals:
            return states
        return states.difference(self.goals)

    def info(self):
        return f"roots: {len(self.roots)}, states: {len(self.states)}, " \
               f"transitions: {self.num_transitions()} ({len(self.optimal_transitions)} optimal)," \
               f" goals: {len(self.goals)}, alive: {len(self.alive_states)}"

    def __str__(self):
        return "TransitionsSample[{}]".format(self.info())

    def get_sorted_state_ids(self):
        return sorted(self.states.keys())


def mark_optimal(goal_states, root_states, parents):
    """ Collect all those transitions that lie on one arbitrary optimal path to the goal """
    optimal_transitions = set()
    for goal in goal_states:
        previous = current = goal

        while current not in root_states:
            # A small trick: node IDs are ordered by depth, so we can pick the min parent ID and know the resulting path
            # will be optimal
            current = min(parents[current])
            optimal_transitions.add((current, previous))  # We're traversing backwards
            previous = current

    return optimal_transitions


def run_backwards_brfs(g, parents, mincosts, minactions):
    queue = deque([g])
    mincosts[g] = 0
    minactions[g] = []

    # Run a breadth-first search backwards from the given goal state g
    while queue:
        cur = queue.popleft()
        curcost = mincosts[cur]

        if cur not in parents:
            # A root of the (original) breadth-first search could have no parents
            continue

        for par in parents[cur]:
            parcost = mincosts.get(par, math.inf)
            if parcost > curcost + 1:
                queue.append(par)
                mincosts[par] = curcost + 1
                minactions[par] = [cur]
            elif parcost == curcost + 1:
                minactions[par].append(cur)


def mark_all_optimal(goals, parents):
    """ Collect all transitions that lie on at least one optimal plan starting from some alive state (i.e. solvable,
     reachable, and not a goal). """
    vstar, minactions = {}, {}
    for g in goals:
        run_backwards_brfs(g, parents, vstar, minactions)

    # minactions contains a map between state IDs and a list with those successors that represent an optimal transition
    # from that state
    optimal_txs = set()
    for s, targets in minactions.items():
        _ = [optimal_txs.add((s, t)) for t in targets]

    # Incidentally, the set of alive states will contain all states with a mincost > 0 and which have been reached on
    # the backwards brfs (i.e. for which mincost[s] is actually defined). Note that the "reachable" part of a state
    # being alive is already guaranteed by the fact that the state is on the sample, as we sample states with a simple
    # breadth first search.
    alive = {s for s, cost in vstar.items() if cost > 0}

    return optimal_txs, alive, vstar


def log_sampled_states(sample, filename):
    optimal_s = set(x for x, _ in sample.optimal_transitions)

    with open(filename, 'w') as f:
        for id_, state in sample.states.items():
            parents = sample.parents.get(id_, [])
            state_parents = ", ".join(sorted(map(str, parents)))

            tx = sorted(sample.transitions[id_])
            state_children = ", ".join(f'{x}{"+" if (id_, x) in sample.optimal_transitions else ""}' for x in tx)
            atoms = ", ".join(str(atom) for atom in state.as_atoms())
            is_goal = "*" if id_ in sample.goals else ""
            is_expanded = "^" if id_ in sample.expanded else ""
            is_alive = "º" if id_ in sample.alive_states else ""
            is_root = "=" if id_ in sample.roots else ""
            is_optimal = "+" if id_ in optimal_s else ""
            print(f"#{id_}{is_root}{is_goal}{is_optimal}{is_expanded}{is_alive}"
                  f"(parents: {state_parents}, children: {state_children}):\n\t{atoms}", file=f)

        print("Symbols:\n*: goal, \n^: expanded, \nº: alive, \n=: root, \n"
              "+: source of some transition marked as optimal", file=f)
    logging.info('Resampled states logged at "{}"'.format(filename))


def sample_generated_states(config, rng):
    logging.info('Loading state space samples...')
    sample, goals_by_instance = read_transitions_from_files(config.sample_files)

    if not config.create_goal_features_automatically and not sample.goals:
        raise RuntimeError("No goal found in the sample - increase number of expanded states!")

    mark_optimal_transitions(sample)
    logging.info(f"Entire sample: {sample.info()}")

    return sample


def mark_optimal_transitions(sample: TransitionSample):
    """ Marks which transitions are optimal in a transition system according to some selection criterion
    such as marking *all* optimal transitions.
     """
    # Mark all transitions that are optimal from some alive state
    # We also mark which states are alive.
    optimal, alive, sample.vstar = mark_all_optimal(sample.goals, sample.parents)
    sample.mark_as_alive(alive)
    sample.mark_as_optimal(optimal)


def compute_parents(transitions):
    """ Return a dictionary mapping state IDs x to a set with the IDs of the parents of x in the sample. """
    parents = defaultdict(set)
    for source, targets in transitions.items():
        for t in targets:
            parents[t].add(source)
    return parents


def normalize_atom_name(name):
    tmp = name.replace('()', '').replace(')', '').replace('(', ',')
    if "=" in tmp:  # We have a functional atom
        tmp = tmp.replace("=", ',')

    return tmp.split(',')


def remap_state_ids(states, goals, transitions, unsolvable, remap):
    new_goals = {remap(x) for x in goals}
    new_unsolvable = {remap(x) for x in unsolvable}
    new_states = OrderedDict()
    for i, s in states.items():
        new_states[remap(i)] = s

    new_transitions = defaultdict(set)
    for source, targets in transitions.items():
        new_transitions[remap(source)] = {remap(t) for t in targets}

    return new_states, new_goals, new_transitions, new_unsolvable


def read_transitions_from_files(filenames):
    assert len(filenames) > 0

    goals_by_instance = []
    sample = TransitionSample()
    for instance_id, filename in enumerate(filenames, 0):
        s, g, transitions, unsolvable = read_single_sample_file(filename)
        assert next(iter(s.keys())) == 0  # Make sure state IDs in the sample file start by 0

        starting_state_id = sample.num_states()
        remap = lambda state: state + starting_state_id
        s, g, transitions, expanded = remap_state_ids(s, g, transitions, unsolvable, remap=remap)
        assert next(iter(s)) == starting_state_id

        sample.add_transitions(s, transitions, instance_id, unsolvable)
        sample.mark_as_goals(g)
        goals_by_instance.append(g)

    return sample, goals_by_instance


def read_single_sample_file(filename):
    state_atoms = {}
    transitions = defaultdict(set)
    transitions_inv = defaultdict(set)
    goal_states = set()
    expanded = set()

    nlines = 0  # The number of useful lines processed
    for line in read_file(filename):
        if line.startswith('(E)'):  # An edge, with format "(E) 5 12"
            pid, cid = (int(x) for x in line[4:].split(' '))
            transitions[pid].add(cid)
            transitions_inv[cid].add(pid)
            nlines += 1

        elif line.startswith('(N)'):  # A node
            # Format "(N) <id> <is_goal> <is_expanded> <space-separated-atom-list>", e.g.:
            # (N) 12
            elems = line[4:].split(' ')
            sid = int(elems[0])
            if int(elems[1]):  # The state is a goal state
                goal_states.add(sid)
            if int(elems[2]):  # The state is a dead-end
                expanded.add(sid)

            state_atoms[sid] = tuple(normalize_atom_name(atom) for atom in elems[3:])
            nlines += 1

    # Make sure all edge IDs have actually been declared as a state
    for src in transitions:
        assert src in state_atoms and all(dst in state_atoms for dst in transitions[src])

    # Make sure that the number of outgoing and incoming edges coincide
    num_tx = sum(len(t) for t in transitions.values())
    assert num_tx == sum(len(t) for t in transitions_inv.values())

    logging.info('%s: #lines-raw-file=%d, #states-by-id=%d, #transition-entries=%d, #transitions=%d' %
                 (filename_core(filename), nlines, len(state_atoms), len(transitions), num_tx))

    ordered = OrderedDict()  # Make sure we return an ordered dictionary
    for id_ in sorted(state_atoms.keys()):
        ordered[id_] = state_atoms[id_]
    return ordered, goal_states, transitions, expanded


def generate_sample(config, rng):
    sample = sample_generated_states(config, rng)
    log_sampled_states(sample, config.resampled_states_filename)
    print_transition_matrix(sample, config.transitions_info_filename)
    return sample


def run(config, data, rng):
    assert not data
    return ExitCode.Success, dict(sample=generate_sample(config, rng))

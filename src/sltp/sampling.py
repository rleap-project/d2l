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
        self.transitions = defaultdict(set)
        self.parents = dict()
        # self.problem = dict()
        self.roots = set()  # The set of all roots
        self.instance_roots = []  # The root of each instance
        self.goals = set()
        self.alive_states = set()
        self.optimal_transitions = set()
        self.expanded = set()
        self.deadends = set()
        self.instance = dict()  # A mapping between states and the problem instances they came from
        self.remapping = dict()
        self.vstar = {}

    def get_state_id(self, state):
        for k, v in self.states.items() :
            if state == v :
                return k
        return -1

    def add_transitions(self, states, transitions, instance_id, deadends):
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
        self.deadends.update(deadends)
        # We consider a state expanded if it has some child or it is marked as a deadend
        self.expanded.update(s for s in states if (s in transitions.keys() and len(transitions[s]) > 0) or s in self.deadends)

    def mark_as_root(self, state):
        self.roots.add(state)
        self.instance_roots.append(state)

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

    def resample(self, selected):
        """ Resample (i.e. project) the current sample into a new sample that will contain only states specified
        in `selected` plus their children. """
        # all_in_sample will contain all states we want to sample plus their children state IDs
        all_in_sample = set(selected)
        for s in selected:  # Add the children
            all_in_sample.update(self.transitions[s])

        # Sort the selected states and compute the remapping function
        ordered_sample = list(sorted(all_in_sample))
        remapping = {x: i for i, x in enumerate(ordered_sample)}

        # Pick the selected elements from the data structures
        goals = {remapping[x] for x in ordered_sample if x in self.goals}
        alive_states = {remapping[x] for x in ordered_sample if x in self.alive_states}
        deadends = {remapping[x] for x in ordered_sample if x in self.deadends}
        roots = {remapping[x] for x in ordered_sample if x in self.roots}
        optimal = {(remapping[x], remapping[y]) for x, y in self.optimal_transitions
                   if x in remapping and y in remapping}

        vstar = {remapping[x]: self.vstar[x] for x in ordered_sample if x in self.vstar}

        instance = dict()
        states = OrderedDict()
        for i, s in self.states.items():  # Iterate following the order in self.states
            if i in remapping:
                states[remapping[i]] = s
                instance[remapping[i]] = self.instance[i]

        transitions = defaultdict(set)
        for source, targets in self.transitions.items():
            if source in selected:
                mapped_source = remapping[source]
                for t in targets:
                    assert t in remapping  # because we have added all children of the selected nodes
                    transitions[mapped_source].add(remapping[t])

        resampled = TransitionSample()
        resampled.add_transitions(states, transitions, 0, deadends)
        resampled.instance = instance
        resampled.mark_as_goals(goals)
        resampled.mark_as_optimal(optimal)
        resampled.mark_as_alive(alive_states)
        resampled.vstar = vstar
        resampled.remapping = remapping
        _ = [resampled.mark_as_root(r) for r in roots]
        return resampled

    def get_one_goal_per_instance(self):
        """ Return the min-id state for each instance that is marked as a goal """
        instance_goals = defaultdict(list)
        for sid in self.goals:
            instance_goals[self.instance[sid]].append(sid)
        return set(min(instance_goals[x]) for x in range(0, len(self.roots)) if instance_goals[x])


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


def print_atom(atom):
    assert len(atom) > 0
    if len(atom) == 1:
        return atom[0]
    return "{}({})".format(atom[0], ", ".join(atom[1:]))


def log_sampled_states(sample, filename):
    optimal_s = set(x for x, _ in sample.optimal_transitions)

    with open(filename, 'w') as f:
        for id_, state in sample.states.items():
            # We need to recompute the parenthood relation with the remapped states to log it!
            parents = sample.parents[id_] if id_ in sample.parents else []
            state_parents = ", ".join(sorted(map(str, parents)))

            tx = sorted(sample.transitions[id_])
            state_children = ", ".join(f'{x}{"+" if (id_, x) in sample.optimal_transitions else ""}' for x in tx)
            atoms = ", ".join(print_atom(atom) for atom in state)
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


def sample_first_x_states(root_states, sample_sizes):
    """ Sample the first sample_sizes[i] states of instance i, i.e., the integers going from root_states[i] to
    root_states[i] + sample_sizes[i] """
    sampled = set()
    assert len(root_states) == len(sample_sizes)
    for root, size in zip(root_states, sample_sizes):
        sampled.update(range(root, root+size))
    return sampled


def sample_generated_states(config, rng):
    logging.info('Loading state space samples...')
    sample, goals_by_instance = read_transitions_from_files(config.sample_files)

    if not config.create_goal_features_automatically and not sample.goals:
        raise RuntimeError("No goal found in the sample - increase number of expanded states!")

    mark_optimal_transitions(sample)
    logging.info(f"Entire sample: {sample.info()}")

    if config.num_sampled_states is not None:
        # Resample the full sample and extract only a few specified states
        selected = sample_first_x_states(sample.instance_roots, config.num_sampled_states)
        states_in_some_optimal_transition = sample.compute_optimal_states(include_goals=True)
        selected.update(states_in_some_optimal_transition)
        sample = sample.resample(set(selected))
        logging.info(f"Sample after resampling: {sample.info()}")

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


def random_sample(config, goal_states, rng, states, transitions, parents):
    num_states = min(len(states), config.num_states)
    assert len(set(config.num_sampled_states)) == 1, "ATM only random sample with fixed sample size"
    sample_size = config.num_sampled_states[0]
    if sample_size > num_states:
        raise RuntimeError(
            "Only {} expanded states, cannot sample {}".format(num_states, sample_size))
    # Although this might fail is some state was a dead-end? In that case we should perhaps revise the code below
    assert num_states == len(transitions)

    selected = sample_expanded_and_goal_states(config.num_sampled_states, goal_states, num_states, parents, rng)
    return selected


def sample_expanded_and_goal_states(sample_size, goal_states, num_states, parents, rng):
    expanded_states = list(range(0, num_states))
    # We will at least select all of those goal states that have been expanded plus one parent of those that not,
    # so that we maximize the number of goal states in the sample.
    # This is not the only possible strategy, and will be problematic if e.g. most states are goal states, but
    # for the moment being we're happy with it
    enforced = set()
    for x in expanded_states:
        if x in goal_states:
            enforced.add(x)
        elif parents[x]:
            enforced.add(next(iter(parents[x])))  # We simply pick one arbitrary parent of the non-expanded goal state
    #
    enforced = set()
    non_enforced = [i for i in expanded_states if i not in enforced]
    rng.shuffle(non_enforced)
    all_shuffled = list(enforced) + non_enforced
    return all_shuffled[:sample_size]


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


def remap_state_ids(states, goals, transitions, deadends, remap):

    new_goals = {remap(x) for x in goals}
    new_deadends = {remap(x) for x in deadends}
    new_states = OrderedDict()
    for i, s in states.items():
        new_states[remap(i)] = s

    new_transitions = defaultdict(set)
    for source, targets in transitions.items():
        new_transitions[remap(source)] = {remap(t) for t in targets}

    return new_states, new_goals, new_transitions, new_deadends


def read_transitions_from_files(filenames):
    assert len(filenames) > 0

    goals_by_instance = []
    sample = TransitionSample()
    for instance_id, filename in enumerate(filenames, 0):
        starting_state_id = sample.num_states()
        s, g, tx, deadends = read_single_sample_file(filename)
        assert next(iter(s.keys())) == 0  # Make sure state IDs in the sample file start by 0
        s, g, tx, deadends = remap_state_ids(s, g, tx, deadends,
                                             remap=lambda state: state + starting_state_id)
        assert next(iter(s)) == starting_state_id

        sample.add_transitions(s, tx, instance_id, deadends)
        sample.mark_as_root(starting_state_id)
        sample.mark_as_goals(g)
        goals_by_instance.append(g)

    return sample, goals_by_instance


def read_single_sample_file(filename):
    state_atoms = {}
    transitions = defaultdict(set)
    transitions_inv = defaultdict(set)
    goal_states = set()
    deadends = set()

    nlines = 0  # The number of useful lines processed
    for line in read_file(filename):
        if line.startswith('(E)'):  # An edge, with format "(E) 5 12"
            pid, cid = (int(x) for x in line[4:].split(' '))
            transitions[pid].add(cid)
            transitions_inv[cid].add(pid)
            nlines += 1

        elif line.startswith('(N)'):  # A node
            # Format "(N) <id> <is_goal> <is_deadend> <space-separated-atom-list>", e.g.:
            # (N) 12
            elems = line[4:].split(' ')
            sid = int(elems[0])
            if int(elems[1]):  # The state is a goal state
                goal_states.add(sid)
            if int(elems[2]):  # The state is a dead-end
                deadends.add(sid)

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
    return ordered, goal_states, transitions, deadends


def generate_sample(config, rng):
    sample = sample_generated_states(config, rng)
    log_sampled_states(sample, config.resampled_states_filename)
    print_transition_matrix(sample, config.transitions_info_filename)
    return sample


def run(config, data, rng):
    assert not data
    return ExitCode.Success, dict(sample=generate_sample(config, rng))

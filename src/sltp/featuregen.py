import logging

import os
import stat
from collections import defaultdict

from tarski.dl import PrimitiveConcept, UniversalConcept, NullaryAtom, NominalConcept, GoalConcept, GoalRole, \
    EmptyConcept, GoalNullaryAtom

from .matrices import NP_FEAT_VALUE_TYPE, cast_feature_value_to_numpy_value, log_feature_denotations
from .models import DLModel
from .features import parse_all_instances, compute_models, InstanceInformation
from .util.command import execute
from .util.serialization import unserialize_feature
from .returncodes import ExitCode


def run(config, data, rng):
    return generate_feature_pool(config, data.sample)


def convert_tuple(universe, name, values):
    return (name, ) + tuple(universe.value(i) for i in (values if isinstance(values, tuple) else (values, )))


def goal_predicate_name(name):
    return "{}_g".format(name)


def process_predicate_atoms(universe, p, extension, atoms):
    if (isinstance(p, PrimitiveConcept) and p.name == 'object') or \
            isinstance(p, (UniversalConcept, EmptyConcept)) or isinstance(p, NominalConcept):
        return

    name = goal_predicate_name(p.name) \
        if isinstance(p, (GoalConcept, GoalRole, GoalNullaryAtom)) else p.name  # HACK HACK HACK

    if isinstance(p, NullaryAtom):
        if extension is True:
            atoms.append((name, ))
    else:
        for point in extension:
            atoms.append(convert_tuple(universe, name, point))


def serialize_dl_model(model: DLModel, info: InstanceInformation):
    serialized = []
    universe = info.universe

    # Add fluent information
    for k, v in model.primitive_denotations.items():
        process_predicate_atoms(universe, k, v, serialized)

    # Add static information
    for k, v in model.statics.items():
        process_predicate_atoms(universe, k, v, serialized)

    return serialized


def serialize_static_info(model: DLModel, info: InstanceInformation):
    serialized = []
    universe = info.universe

    # Add fluent information
    for k, v in model.primitive_denotations.items():
        for point in v:
            serialized.append(convert_tuple(universe, k.name, point))

    # Add static information
    for k, v in model.statics.items():
        if (isinstance(k, PrimitiveConcept) and k.name == 'object') or isinstance(k, UniversalConcept):
            continue

        for point in v:
            serialized.append(convert_tuple(universe, k.name, point))

    return serialized


def print_sample_info(sample, infos, model_cache, all_predicates, all_functions, nominals,
                      all_objects, goal_predicate_info, config):
    workspace = config.experiment_dir

    state_info = []
    atoms_per_instance = defaultdict(set)
    # Iterate over all states and collect the necessary information
    for expected_id, (id_, state) in enumerate(sample.states.items(), 0):
        # Check all ids are consecutive, as expected
        assert expected_id == id_
        instance_id = sample.instance[id_]  # The instance to which the state belongs
        full_state = serialize_dl_model(model_cache.models[id_], infos[instance_id])
        atoms_per_instance[instance_id].update(full_state)
        state_info.append([str(instance_id)] + [",".join(atom) for atom in full_state])

    sample_fn = os.path.join(workspace, "sample.io")
    logging.info(f"Printing sample information to '{sample_fn}'")
    with open(sample_fn, "w") as f:
        # First line: all predicate and function names (note diff treatment of their arity)
        predfuns = [(name, ar) for name, ar in all_predicates] + [(name, ar + 1) for name, ar in all_functions]
        print(" ".join("{}/{}".format(name, arity) for name, arity in sorted(predfuns)), file=f)

        # Next: List of all predicates and functions mentioned in the goal
        # Next line disables the "in-goal" functionality that enforces goal-identifying features of the form
        # p_g - p = 0, for all predicates p
        if config.create_goal_features_automatically:
            print(" ".join(name for name, arity in sorted(goal_predicate_info)), file=f)
        else:
            print("", file=f)

        # Next: per-instance information.
        print(len(infos), file=f)  # Number of instances

        assert len(infos) == len(all_objects) == len(atoms_per_instance)
        for instance_id, objects in enumerate(all_objects, start=0):
            print(" ".join(sorted(objects)), file=f)  # all object names in instance i

            # all possible atoms in instance i:
            print("\t".join(",".join(atom) for atom in sorted(atoms_per_instance[instance_id])), file=f)

        # Next: all states. One state per line. first column is instance_id of state, rest are all atoms in that state,
        # including static atoms and type-predicate atoms
        for stinfo in state_info:
            # print one line per state with all state atoms, e.g. at,bob,shed   at,spanner,location;
            print("\t".join(stinfo), file=f)

    nominals_fn = os.path.join(workspace, "nominals.io")
    if nominals:
        logging.info(f"Printing information on nominal concepts to {nominals_fn}")
        with open(nominals_fn, "w") as f:
            # Print off the desired nominals
            print(" ".join(name for name in sorted(x.symbol for x in nominals)), file=f)
    else:
        open(nominals_fn, 'w').close()  # Just write an empty file


def transform_generator_output(config, sample, matrix_filename, info_filename):
    logging.info("Transforming generator output to numpy format...")
    import numpy as np

    logging.info("Reading feature information from {}".format(info_filename))
    with open(info_filename, "r") as f:
        # Line  # 1: feature names
        names = f.readline().rstrip().split("\t")

        # Line #2: feature complexities
        complexities = [int(x) for x in f.readline().rstrip().split("\t")]

        # Line #3: feature types (0: boolean; 1: numeric)
        types = [int(x) for x in f.readline().rstrip().split("\t")]

        # Line #4: whether feature is goal feature (0: No; 1: yes)
        in_goal_features = [bool(int(x)) for x in f.readline().rstrip().split("\t")]
        in_goal_features = set(i for i, v in enumerate(in_goal_features, 0) if v)  # Transform into a set of indexes

    num_features = len(names)
    assert num_features == len(complexities) == len(types)
    num_numeric = sum(types)
    num_binary = num_features - num_numeric
    logging.info("Read {} features: {} numeric + {} binary".format(num_features, num_numeric, num_binary))

    logging.info("Reading denotation matrix from {}".format(matrix_filename))
    # Convert features to a numpy array with n rows and m columns, where n=num states, m=num features

    with open(matrix_filename, "r") as f:
        num_states = sum(1 for _ in f)

    matrix = np.empty(shape=(num_states, num_features), dtype=NP_FEAT_VALUE_TYPE)

    with open(matrix_filename, "r") as f:
        # One line per state with the numeric denotation of all features
        for i, line in enumerate(f, start=0):
            data = line.rstrip().split(' ')
            matrix[i] = [cast_feature_value_to_numpy_value(int(x)) for x in data]

    logging.info("Read denotation matrix with dimensions {}".format(matrix.shape))
    print_actual_output(sample, config, names, complexities, matrix)
    return in_goal_features, len(names)


def generate_output_from_handcrafted_features(sample, config, features, model_cache):
    import numpy as np
    num_features = len(features)
    num_states = len(sample.states)

    names = [str(f) for f in features]
    complexities = [f.complexity() for f in features]
    matrix = np.empty(shape=(num_states, num_features), dtype=NP_FEAT_VALUE_TYPE)

    for i, (sid, atoms) in enumerate(sample.states.items(), start=0):
        assert i == sid
        model = model_cache.get_feature_model(sid)
        denotations = [model.denotation(f) for f in features]
        matrix[i] = [cast_feature_value_to_numpy_value(int(x)) for x in denotations]

    # These next 3 lines just to print the denotation of all features
    if config.print_denotations:
        state_ids = sample.get_sorted_state_ids()
        models = {sid: model_cache.get_feature_model(sid) for sid in sample.states}
        log_feature_denotations(state_ids, features, models, config.feature_denotation_filename, None)

    print_actual_output(sample, config, names, complexities, matrix)
    return [], len(names)


def print_actual_output(sample, config, names, complexities, matrix):
    state_ids = sample.get_sorted_state_ids()
    print_feature_matrix(config.feature_matrix_filename, matrix, state_ids, sample.goals, sample.deadends, names, complexities)


def generate_debug_scripts(target_dir, exe, arguments):
    # If generating a debug build, create some debug script helpers
    shebang = "#!/usr/bin/env bash"
    args = ' '.join(arguments)
    debug_script = "{}\n\n cgdb -ex=run --args {} {}".format(shebang, exe, args)
    memleaks = "{}\n\n valgrind --leak-check=full --show-leak-kinds=all --num-callers=50 --track-origins=yes " \
               "--log-file=\"valgrind-output.$(date '+%H%M%S').txt\" {} {}"\
        .format(shebang, exe, args)

    memprofile = "{}\n\n valgrind --tool=massif {} {}".format(shebang, exe, args)

    make_script(os.path.join(target_dir, 'debug.sh'), debug_script)
    make_script(os.path.join(target_dir, 'memleaks.sh'), memleaks)
    make_script(os.path.join(target_dir, 'memprofile.sh'), memprofile)


def make_script(filename, code):
    with open(filename, 'w') as f:
        print(code, file=f)
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)


def generate_feature_pool(config, sample):
    logging.info(f"Starting generation of feature pool. State sample used to detect redundancies: {sample.info()}")

    parsed_problems = parse_all_instances(config.domain, config.instances)  # Parse all problem instances

    language, nominals, model_cache, infos, all_goal_predicates = compute_models(
        config.domain, sample, parsed_problems, config.parameter_generator)

    all_objects = []
    all_predicates, all_functions = set(), set()
    goal_predicate_info = set()
    for _, lang, _ in parsed_problems:
        all_objects.append(set(c.symbol for c in lang.constants()))
        all_predicates.update((p.name, p.arity) for p in lang.predicates if not p.builtin)
        all_functions.update((p.name, p.arity) for p in lang.functions if not p.builtin)

        # Add goal predicates and functions
        goal_predicate_info = set((goal_predicate_name(p.name), p.uniform_arity())
                                  for p in lang.predicates if not p.builtin and p.name in all_goal_predicates)
        goal_predicate_info.update((goal_predicate_name(f.name), f.uniform_arity())
                                   for f in lang.functions if not f.builtin and f.name in all_goal_predicates)
        all_predicates.update(goal_predicate_info)

        # Add type predicates
        all_predicates.update((p.name, 1) for p in lang.sorts if not p.builtin and p != lang.Object)

    # Write out all input data for the C++ feature generator code
    print_sample_info(sample, infos, model_cache, all_predicates, all_functions, nominals,
                      all_objects, goal_predicate_info, config)

    # If user provides handcrafted features, no need to go further than here
    if config.feature_generator is not None:
        features = deal_with_serialized_features(language, config.feature_generator, config.serialized_feature_filename)
        generate_output_from_handcrafted_features(sample, config, features, model_cache)
        return ExitCode.Success, dict(enforced_feature_idxs=[], in_goal_features=[],
                                      model_cache=model_cache)

    if invoke_cpp_generator(config) != 0:
        return ExitCode.FeatureGenerationUnknownError, dict()

    # Read off the output of the module and transform it into the numpy matrices to be consumed
    # by the next pipeline step
    in_goal_features, nfeatures = transform_generator_output(
        config, sample,
        os.path.join(config.experiment_dir, "feature-matrix.io"),
        os.path.join(config.experiment_dir, "feature-info.io"),)

    return ExitCode.Success, dict(enforced_feature_idxs=[],
                                  in_goal_features=in_goal_features,
                                  model_cache=model_cache)


def invoke_cpp_generator(config):
    logging.info('Invoking C++ feature generation module'.format())
    cmd = os.path.realpath(os.path.join(config.generators_path, "featuregen"))
    args = f" --complexity-bound {config.max_concept_size}" \
           + f" --timeout {config.concept_generation_timeout}" \
           + f" --dist-complexity-bound {config.distance_feature_max_complexity}" \
           + f" --cond-complexity-bound {config.cond_feature_max_complexity}" \
           + (f" --comparison-features" if config.comparison_features else "") \
           + (f" --generate-goal-concepts" if config.generate_goal_concepts else "") \
           + (f" --print-denotations" if config.print_denotations else "") \
           + f" --workspace {config.experiment_dir}"
    args = args.split()
    generate_debug_scripts(config.experiment_dir, cmd, args)
    retcode = execute([cmd] + args)
    return retcode


def deal_with_serialized_features(language, feature_generator, serialized_feature_filename):
    logging.info('Skipping automatic feature generation: User provided set of handcrafted features')
    features = feature_generator(language)
    if features and isinstance(features[0], str):  # Features given as strings, unserialize them
        features = [unserialize_feature(language, f) for f in features]

    # Print the features to the appropriate place to be unserialized later on
    with open(serialized_feature_filename, 'w') as file:
        for f in features:
            print("{}\t{}".format(f, f.complexity()), file=file)
    return features


def print_feature_matrix(filename, matrix, state_ids, goals, deadends, names, complexities):
    ngoals, nfeatures = len(goals), len(names)
    assert nfeatures == len(complexities) == matrix.shape[1]
    logging.info("Printing matrix of {} features x {} states to '{}'".format(nfeatures, len(state_ids), filename))

    with open(filename, 'w') as f:
        # Header row: <#states> <#features> <#goals>
        print("{} {} {}".format(len(state_ids), nfeatures, ngoals), file=f)

        # Line #2:: <list of feature names>
        print("{}".format(" ".join(name.replace(" ", "") for name in names)), file=f)

        # Line #3: <list of feature costs>
        print("{}".format(" ".join(str(c) for c in complexities)), file=f)

        # Line #4: <list of goal state IDs>
        print("{}".format(" ".join(map(str, goals))), file=f)

        # Line #5: <list of expanded state IDs>
        print("{}".format(" ".join(map(str, deadends))), file=f)

        # next lines: one per each state with format: <state-index> <#features-in-state> <list-features>
        # each feature has format: <feature-index>:<value>
        for s in state_ids:
            feature_values = ((i, matrix[s][i]) for i in range(0, nfeatures))
            nonzero_features = [(i, v) for i, v in feature_values if v != 0]
            flist = " ".join(f"{i}:{int(v)}" for i, v in nonzero_features)
            print(f"{s} {len(nonzero_features)} {flist}", file=f)

import logging
import os

from tarski.dl import FeatureValueChange

from sltp.separation import TransitionClassificationPolicy, DNFAtom

from .util.tools import load_selected_features, IdentifiedFeature
from .util.command import execute, read_file
from .returncodes import ExitCode


def invoke_cpp_module(config, data, validate_features=None):
    logging.info('Calling C++ MaxSAT module')
    cmd = os.path.realpath(os.path.join(config.generators_path, "cnfgen"))
    args = ["--workspace", config.experiment_dir]
    args += ["--validate-features", ",".join(map(str, validate_features))] if validate_features is not None else []
    args += ["--use-equivalence-classes"] if config.use_equivalence_classes else []
    args += ["--use-feature-dominance"] if config.use_feature_dominance else []
    args += ["--v_slack", str(config.v_slack)]
    args += ["--distinguish-goals"] if config.distinguish_goals else []
    args += ["--initial-sample-size", str(config.initial_sample_size)]
    args += ["--refinement-batch-size", str(config.refinement_batch_size)]
    args += ["--seed", str(config.seed)]
    args += ["--verbosity", str(config.verbosity)]
    args += ["--acyclicity", str(config.acyclicity)]
    args += ["--encodings_dir", str(config.encodings_dir)]
    args += ["--sampling_strategy", str(config.sampling_strategy)]
    retcode = execute([cmd] + args)

    return {  # Let's map the numeric code returned by the c++ app into an ExitCode object
        0: ExitCode.Success,
        1: ExitCode.MaxsatModelUnsat,
        2: ExitCode.IterativeMaxsatApproachSuccessful
    }.get(retcode, ExitCode.CNFGenerationUnknownError)


def run(config, data, rng):
    exitcode = invoke_cpp_module(config, data)
    if exitcode != ExitCode.Success:
        return exitcode, dict(d2l_policy=None)

    # Parse the DNF transition-classifier and transform it into a policy
    policy = parse_dnf_policy(config)

    policy.minimize()
    print("Final Policy:")
    policy.print_aaai20()

    return exitcode, dict(d2l_policy=policy)


def parse_dnf_policy(config):
    fval_map = {
        "=0": False,
        ">0": True,
        "INC": FeatureValueChange.INC,
        "DEC": FeatureValueChange.DEC,
        "NIL": FeatureValueChange.NIL,
    }
    language = config.language_creator(config)
    policy = None
    fmap = {}
    for i, line in enumerate(read_file(config.experiment_dir + "/classifier.dnf"), 0):
        if i == 0:  # First line contains feature IDs only
            fids = list(map(int, line.split()))
            fs = load_selected_features(language, fids, config.serialized_feature_filename)
            fmap = {i: IdentifiedFeature(f, i, config.feature_namer(str(f))) for i, f in zip(fids, fs)}
            policy = TransitionClassificationPolicy(list(fmap.values()))
            continue

        clause = []
        for lit in line.split(', '):
            f, val = lit.split(' ')
            fid = int(f[2:-1])
            clause.append(DNFAtom(fmap[fid], fval_map[val]))
        policy.add_clause(frozenset(clause))
    return policy

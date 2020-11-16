import importlib
import os
import sys

from ..util import console
from ..util.bootstrap import setup_argparser
from ..util.defaults import generate_experiment


def import_from_file(filename):
    """ Import a module from a given file path """
    import importlib.util
    spec = importlib.util.spec_from_file_location("imported", filename)
    if spec is None:
        report_and_exit(f'Could not import Python module "{filename}"')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_experiment_file(filename):
    """ Import a module from a given file path """
    if os.path.isfile(filename):
        return import_from_file(filename)

    try:
        return importlib.import_module(filename)
    except ImportError:
        report_and_exit(f'No script named "{filename}.py" found on current directory')


def report_and_exit(msg):
    print(f"ERROR: {msg}")
    sys.exit(-1)


def do(expid, steps=None, workspace=None, show_steps_only=False):
    name_parts = expid.split(":")
    if len(name_parts) != 2:
        report_and_exit(f'Wrong experiment ID syntax "{expid}". Expected format <domain>:<experiment_name>')

    scriptname, expname = name_parts
    mod = import_experiment_file(scriptname)

    experiments = None
    try:
        experiments = mod.experiments()
    except AttributeError:
        report_and_exit(f'Expected method "experiments" not found in script "{scriptname}"')

    if expname not in experiments:
        report_and_exit(f'No experiment named "{expname}" in current experiment script')

    parameters = experiments[expname]
    if workspace is not None:
        parameters["workspace"] = workspace

    experiment = generate_experiment(expid, **parameters)

    if show_steps_only:
        console.print_hello()
        print(f'Experiment with id "{expid}" is configured with the following steps:')
        print(experiment.print_description())
        return

    experiment.run(steps)


def run():
    args = setup_argparser().parse_args(sys.argv[1:])
    do(args.exp_id, args.steps, args.workspace, args.show)

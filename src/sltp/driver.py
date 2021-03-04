import copy
import logging
import multiprocessing
import os
import resource
import sys

from .returncodes import ExitCode
from .errors import CriticalPipelineError
from .util import console
from .util.naming import compute_serialization_name
from .util.serialization import deserialize, serialize
from .util import performance


BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
BENCHMARK_DIR = os.path.join(BASEDIR, 'domains')
SAMPLE_DIR = os.path.join(BASEDIR, 'samples')


class InvalidConfigParameter(Exception):
    def __init__(self, msg=None):
        msg = msg or 'Invalid configuration parameter'
        super().__init__(msg)


def get_step(steps, step_id):
    """*step_name* can be a step's name or number."""
    assert isinstance(step_id, int)
    try:
        return steps[step_id - 1]
    except IndexError:
        logging.critical('There is no step number {}'.format(step_id))


def check_int_parameter(config, name, positive=False):
    try:
        config[name] = int(config[name])
        if positive and config[name] <= 0:
            raise ValueError()
    except ValueError:
        raise InvalidConfigParameter('Parameter "{}" must be a {} integer value'.format(
            name, "positive " if positive else ""))


class Step:
    def __init__(self, **kwargs):
        self.config = self.process_config(self.parse_config(**kwargs))

    def process_config(self, config):
        return config  # By default, we do nothing

    def get_required_attributes(self):
        raise NotImplementedError()

    def get_required_data(self):
        raise NotImplementedError()

    def parse_config(self, **kwargs):
        config = copy.deepcopy(kwargs)
        for attribute in self.get_required_attributes():
            if attribute not in kwargs:
                raise RuntimeError(f'Missing configuration parameter "{attribute}" in pipeline step "{self.__class__}"')
            config[attribute] = kwargs[attribute]

        return config

    def description(self):
        raise NotImplementedError()

    def get_step_runner(self):
        raise NotImplementedError()


def save(basedir, output):
    if not output:
        return

    def serializer():
        return tuple(serialize(data, compute_serialization_name(basedir, name)) for name, data in output.items())

    console.log_time(serializer, logging.DEBUG,
                     'Serializing data elements "{}" to directory "{}"'.format(', '.join(output.keys()), basedir))


def _deserializer(basedir, items):
    return dict((k, deserialize(compute_serialization_name(basedir, k))) for k in items)


def load(basedir, items):
    def deserializer():
        return _deserializer(basedir, items)

    output = console.log_time(deserializer, logging.DEBUG,
                              'Deserializing data elements "{}" from directory "{}"'.format(', '.join(items), basedir))
    return output


class StepRunner:
    """ Run the given step """
    def __init__(self, stepnum, step_name, target, required_data):
        self.start = self.elapsed_time()
        self.target = target
        self.stepnum = stepnum
        self.step_name = step_name
        self.required_data = required_data
        self.loglevel = None

    def elapsed_time(self):
        info_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        info_self = resource.getrusage(resource.RUSAGE_SELF)
        # print("({}) Self: {}".format(os.getpid(), info_self))
        # print("({}) Children: {}".format(os.getpid(), info_children))
        return info_children.ru_utime + info_children.ru_stime + info_self.ru_utime + info_self.ru_stime

    def used_memory(self):
        return performance.memory_usage()

    def setup(self, quiet):
        self.loglevel = logging.getLogger().getEffectiveLevel()
        if quiet:
            logging.getLogger().setLevel(logging.ERROR)
        else:
            print(console.header("(pid: {}) STARTING STEP #{}: {}".format(os.getpid(), self.stepnum, self.step_name)))

    def teardown(self, quiet):
        if quiet:
            logging.getLogger().setLevel(self.loglevel)
        else:
            current = self.elapsed_time()
            print(console.header("END OF STEP #{}: {}. {:.2f} CPU sec - {:.2f} MB".format(
                self.stepnum, self.step_name, current - self.start, self.used_memory())))

    def run(self, config):
        exitcode = self._run(config)
        self.teardown(config.quiet)
        return exitcode

    def _run(self, config):
        """ Run the StepRunner target.
            This method will also be the entry point of spawned subprocess in the case
            of SubprocessStepRunners
        """
        self.setup(config.quiet)

        data = Bunch(load(config.experiment_dir, self.required_data)) if self.required_data else None
        # The old strategy based on numpy.random.RandomState objects was problematic, first because
        # the object seems to be in deprecation (https://numpy.org/doc/stable/reference/random/legacy.html),
        # and also because we allow the user to run subsets of pipeline steps, in which case the sequence
        # of random numbers that she will get will not always be the same. To illustrate, if I run steps 1 and 2,
        # both invoking the RNG, step 2 will not see the same sequence of random numbers than if I run step 2 in
        # isolation, because the RNG is seeded once at the beginning. A better strategy would be to store the
        # RNG state between pipeline steps.
        # However, because at the moment we're not using the RNG from Python, I'm leaving the implementation of all
        # this to the future, if necessary.
        rng = None

        try:
            exitcode, output = self.target(config=config, data=data, rng=rng)
        except Exception as exception:
            # Flatten the exception so that we make sure it can be serialized,
            # and return it immediately so that it can be reported from the parent process
            logging.error("Critical error in the pipeline")
            import traceback
            traceback.print_exception(None, exception, exception.__traceback__)
            raise CriticalPipelineError("Error: {}".format(str(exception)))

        save(config.experiment_dir, output)
        # profiling.start()
        return exitcode


class SubprocessStepRunner(StepRunner):
    """ Run the given step by spawning a subprocess and waiting for its finalization """
    def _run(self, config):
        pool = multiprocessing.Pool(processes=1)
        result = pool.apply_async(StepRunner._run, (), {"self": self, "config": config})
        exitcode = result.get()
        pool.close()
        pool.join()
        return exitcode

    def used_memory(self):
        info_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        # print("({}) Children: {}".format(os.getpid(), info_children))
        return info_children.ru_maxrss / 1024.0


class Experiment:
    def __init__(self, all_steps, parameters):
        self.all_steps = all_steps

    def print_description(self):
        return "\t" + "\n\t".join("{}. {}".format(i, s.description()) for i, s in enumerate(self.all_steps, 1))

    def run(self, steps=None):
        console.print_hello()
        # If no steps were given on the commandline, run all exp steps.
        selected = self.all_steps if not steps else [get_step(self.all_steps, step_id) for step_id in steps]

        for stepnum, step in enumerate(selected, start=1):
            try:
                run_and_check_output(step, stepnum, SubprocessStepRunner)
            except Exception as e:
                logging.error((step, _create_exception_msg(step, e)))
                return e
        return ExitCode.Success


def run_and_check_output(step, stepnum, runner_class, raise_on_error=True):
    runner = runner_class(stepnum, step.description(), step.get_step_runner(), step.get_required_data())
    exitcode = runner.run(config=Bunch(step.config))
    if raise_on_error and exitcode is not ExitCode.Success:
        raise RuntimeError(_create_exception_msg(step, exitcode))
    return exitcode


def _create_exception_msg(step, e):
    return 'Critical error processing step "{}". Error message: {}'.\
        format(step.description(), e)


def run_experiment(experiment, argv):
    retcode = experiment.run(argv)
    if retcode != ExitCode.Success:
        sys.exit(-1)
    

class Bunch:
    def __init__(self, adict):
        self.__dict__.update(adict)

    def to_dict(self):
        return self.__dict__.copy()

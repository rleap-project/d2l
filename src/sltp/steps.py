import os
import sys

from .returncodes import ExitCode
from .util.command import execute
from .driver import Step, InvalidConfigParameter, check_int_parameter
from .util.naming import compute_sample_filenames, compute_test_sample_filenames, compute_info_filename, \
    compute_maxsat_filename


class PlannerStep(Step):
    """ Run some planner on certain instance(s) to get the sample of transitions """
    VALID_DRIVERS = ("bfs", )

    def get_required_attributes(self):
        return ["instances", "domain", "num_states", "planner_location", "driver", "test_instances",
                "num_tested_states"]

    def get_required_data(self):
        return []

    def process_config(self, config):
        if config["driver"] not in self.VALID_DRIVERS:
            raise InvalidConfigParameter(f'"driver" must be one of: {self.VALID_DRIVERS}')

        for i in config["instances"]:
            if not os.path.isfile(i):
                raise InvalidConfigParameter(f'"instances" contains non-existing path "{config["instances"]}"')

        if not os.path.isfile(config["domain"]):
            raise InvalidConfigParameter(f'Specified domain file "{config["domain"]}" does not exist')

        if not os.path.isdir(config["planner_location"]):
            raise InvalidConfigParameter(f'Specified planner location "{config["planner_location"]}" does not exist')

        config["sample_files"] = compute_sample_filenames(**config)
        config["test_sample_files"] = compute_test_sample_filenames(**config)
        return config

    def description(self):
        return "Sampling of the state space"

    def get_step_runner(self):
        return _run_brfs


class TransitionSamplingStep(Step):
    """ Generate the sample of transitions from the set of solved planning instances """
    def get_required_attributes(self):
        return ["sample_files", "experiment_dir"]

    def get_required_data(self):
        return []

    def process_config(self, config):
        config["resampled_states_filename"] = os.path.join(config["experiment_dir"], 'sample.txt')
        config["transitions_info_filename"] = compute_info_filename(config, "transitions-info.io")

        ns = config["num_sampled_states"]
        if ns is not None:
            if isinstance(ns, int):
                ns = [ns]

            if len(config["instances"]) != len(ns):
                if len(ns) == 1:
                    ns = ns * len(config["instances"])
                else:
                    raise InvalidConfigParameter('"num_sampled_states" should have same length as "instances"')
            config["num_sampled_states"] = ns

        if config["sampling"] == "random" and config["num_sampled_states"] is None:
            raise InvalidConfigParameter('sampling="random" requires that option "num_sampled_states" is set')

        return config

    def description(self):
        return "Generation of the training sample"

    def get_step_runner(self):
        from . import sampling
        return sampling.run


class CPPFeatureGenerationStep(Step):
    """ Generate exhaustively a set of all features up to a given complexity from the transition (state) sample """
    def get_required_attributes(self):
        return ["domain", "experiment_dir", "max_concept_size", "concept_generation_timeout"]

    def get_required_data(self):
        return ["sample"]

    def process_config(self, config):
        check_int_parameter(config, "max_concept_size")

        config["parameter_generator"] = config.get("parameter_generator", None)
        config["concept_denotation_filename"] = compute_info_filename(config, "concept-denotations.txt")
        config["feature_denotation_filename"] = compute_info_filename(config, "feature-denotations.txt")
        config["serialized_feature_filename"] = compute_info_filename(config, "serialized-features.io")

        return config

    def description(self):
        return "C++ feature generation module"

    def get_step_runner(self):
        from . import featuregen
        return featuregen.run


class CPPMaxsatProblemGenerationStep(Step):
    """ Generate the standard SLTP Max-sat CNF encoding """
    def get_required_attributes(self):
        return ["experiment_dir"]

    def get_required_data(self):
        return ["model_cache"]

    def process_config(self, config):
        config["top_filename"] = compute_info_filename(config, "top.dat")
        config["cnf_filename"] = compute_maxsat_filename(config)
        config["good_transitions_filename"] = compute_info_filename(config, "good_transitions.io")
        config["good_features_filename"] = compute_info_filename(config, "good_features.io")
        config["wsat_varmap_filename"] = compute_info_filename(config, "varmap.wsat")
        config["wsat_allvars_filename"] = compute_info_filename(config, "allvars.wsat")
        return config

    def description(self):
        return "C++ CNF generation module"

    def get_step_runner(self):
        from . import cnfgen
        return cnfgen.run


class D2LPolicyTestingStep(Step):
    """  """
    def get_required_attributes(self):
        return ["experiment_dir", "test_instances", "test_domain"]

    def process_config(self, config):
        if config["test_domain"] is not None:
            if not os.path.isfile(config["test_domain"]):
                raise InvalidConfigParameter('"test_domain" must be either None or the path to an existing domain file')
            if any(not os.path.isfile(i) for i in config["test_instances"]):
                raise InvalidConfigParameter('"test_instances" must point to existing files')

        return config

    def get_required_data(self):
        return ["d2l_policy"]

    def description(self):
        return "Testing of the transition-classification policy"

    def get_step_runner(self):
        from . import tester
        return tester.test_d2l_policy


def _run_brfs(config, data, rng):
    # Run the planner on all training and test instances
    def run(d, i, o, num_states):
        if num_states == "until_first_goal":
            num_string = "until_first_goal=true"
        elif num_states == "all":
            num_string = "max_expansions=-1"
        else:
            assert isinstance(num_states, int)
            num_string = f"max_expansions={num_states}"
        params = f'-i {i} --domain {d} --driver {config.driver} --workspace={config.experiment_dir}/planner --options={num_string}'
        execute(command=[sys.executable, "run.py"] + params.split(' '), stdout=o, cwd=config.planner_location)

    for i, o in zip(config.instances, config.sample_files):
        run(config.domain, i, o, config.num_states)

    for i, o in zip(config.test_instances, config.test_sample_files):
        if config.num_tested_states > 0:
            run(config.test_domain, i, o, config.num_tested_states)

    return ExitCode.Success, dict()


def generate_pipeline(pipeline, **kwargs):
    pipeline = DEFAULT_PIPELINES[pipeline] if isinstance(pipeline, str) else pipeline
    pipeline, config = generate_pipeline_from_list(pipeline, **kwargs)
    return pipeline


def generate_pipeline_from_list(elements, **kwargs):
    steps = []
    config = kwargs
    for klass in elements:
        step = klass(**config)
        config = step.config
        steps.append(step)
    return steps, config


DEFAULT_PIPELINES = dict(
    d2l_pipeline=[
        PlannerStep,
        TransitionSamplingStep,
        CPPFeatureGenerationStep,
        CPPMaxsatProblemGenerationStep,
        D2LPolicyTestingStep,
    ],
)

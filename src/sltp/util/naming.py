import os


def filename_core(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def compute_sample_filename(instance):
    return f"samples_{filename_core(instance)}.txt"


def compute_sample_filenames(experiment_dir, instances, **_):
    return [os.path.join(experiment_dir, compute_sample_filename(i)) for i in instances]


def compute_test_sample_filenames(experiment_dir, test_instances, **_):
    return [os.path.join(experiment_dir, compute_sample_filename(i)) for i in test_instances]


def compute_serialization_name(basedir, name):
    return os.path.join(basedir, f'{name}.pickle')


def compute_maxsat_filename(config):
    return compute_info_filename(config, "theory.wsat")


def compute_info_filename(config, name):
    return os.path.join(config["experiment_dir"], name)

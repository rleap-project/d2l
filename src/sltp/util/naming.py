import os


def filename_core(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def compute_sample_filename(instance, width):
    return "samples_{}_w{}.txt".format(filename_core(instance), width)


def compute_sample_filenames(experiment_dir, instances, max_width, **_):
    return [os.path.join(experiment_dir, compute_sample_filename(i, w)) for i, w in zip(instances, max_width)]


def compute_test_sample_filenames(experiment_dir, test_instances, **_):
    return [os.path.join(experiment_dir, compute_sample_filename(i, -1)) for i in test_instances]


def compute_serialization_name(basedir, name):
    return os.path.join(basedir, f'{name}.pickle')


def compute_maxsat_filename(config):
    return compute_info_filename(config, "theory.wsat")


def compute_maxsat_variables_filename(config):
    return compute_info_filename(config, "maxsat_variables.txt")


def compute_info_filename(config, name):
    return os.path.join(config["experiment_dir"], name)

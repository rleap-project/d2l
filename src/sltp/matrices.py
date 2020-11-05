""" Formatting and printing of the output of the concept-based feature generation process,
along with some other related output necessary for subsequent steps in the pipeline """
import itertools
import logging
import math
import sys

import numpy as np


PRUNE_DUPLICATE_FEATURES = True
NP_FEAT_VALUE_TYPE = np.int8  # Keep it allowing negative values, so that we can subtract without overflow!


def next_power_of_two(x):
    """ Return the smallest power of two Z such that Z >= x """
    if x == 0:
        return 0
    return 2 ** (math.ceil(math.log2(x)))


def cast_feature_value_to_numpy_value(value):
    """ Cast a given feature value into a suitable numpy value, if possible, or raise error if not """
    assert value >= 0
    max_ = np.iinfo(NP_FEAT_VALUE_TYPE).max
    if value == sys.maxsize or value == 2147483647:  # std::numeric_limits<int>::max(). Yes, this is not portable :-)
        return max_

    if value >= max_:  # Max value reserved to denote infty.
        raise RuntimeError("Cannot cast feature value {} into numpy value".format(value))

    return value


def print_feature_info(config, features):
    filename = config.feature_info_filename
    logging.info("Printing feature info for {} features to '{}'".format(len(features), filename))

    with open(filename, 'w') as f:
        for feat in features:
            print("{} {}".format(feat, feat.complexity()), file=f)


def log_feature_denotations(state_ids, features, models, feature_denotation_filename, selected=None):
    selected = selected or features
    selected = ((str(f), f) for f in selected)
    selected = sorted(selected, key=lambda x: x[0])  # Sort features by name

    with open(feature_denotation_filename, 'w') as file:
        for s, (fname, f) in itertools.product(state_ids, selected):
            val = models[s].denotation(f)
            print("s_{}[{}] = {}".format(s, fname, val), file=file)
    logging.info("Logged feature denotations at '{}'".format(feature_denotation_filename))


def printer(feature, value):
    return "1" if feature.bool_value(value) else "0"


def int_printer(value):
    return str(int(value))

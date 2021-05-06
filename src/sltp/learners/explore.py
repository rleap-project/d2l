import argparse
import os
import random
from collections import defaultdict

import numpy as np
import sympy
# from sklearn import linear_model
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.preprocessing import PolynomialFeatures
from tarski.utils import resources

from .utils import load, detect_boolean, Dataset, describe


def filter_by_col_idx(data, idxs):
    matrix = data.matrix[:, idxs]
    names = data.names[idxs]
    complexities = data.complexities[idxs]
    return Dataset(names, complexities, matrix, data.labels)


def filter_by_row_idx(data, idxs):
    matrix = data.matrix[idxs, :]
    labels = data.labels[idxs]
    return Dataset(data.names, data.complexities, matrix, labels)


def filter_by_complexity(data, k):
    idxs_under_k, = np.where(data.complexities <= k)
    data = filter_by_col_idx(data, idxs_under_k)
    print(f"Considering {data.numfeatures()} features under complexity {k}.")
    return data


def filter_by_names(data, names):
    names = set(n for n in names)
    data = filter_by_col_idx(data, idxs=[i for i in range(data.matrix.shape[1]) if data.names[i] in names])
    print(f"Considering {data.numfeatures()} features with the given inclusion pattern.")
    return data


def remove_linear_combinations(data):
    with resources.timing("Removing linear combinations of features", newline=False):
        _, inds = sympy.Matrix(data.matrix).rref()
        data = filter_by_col_idx(data, list(inds))
    print(f"Considering {data.numfeatures()} features after removing linear combinations.")
    return data


# clf = linear_model.Lasso(alpha=0.2, max_iter=10000)
# print("\n\nLasso\n\n")
# for k in range(4, 11):
#     idxs_under_k, = np.where(complexities <= k)
#     relevant = matrix[:, idxs_under_k]
#     print(f"Fitting {relevant.shape} feature matrix to {labels.shape} labels")
#
#     clf.fit(relevant, labels)
#
#     print_function(names, clf)
#
#     score = clf.score(relevant, labels)
#     print(f"Predictor score with k={k}: {score}")
#     # to_predict = relevant[predictidxs]
#     # predictions = clf.predict(to_predict)
#     for i in sample:
#         print(f"h*(s_{i}) = {labels[i]}, vs h(s_{i}) = {clf.predict(relevant[i].reshape(1, -1))}")


def filter_features(data, max_k=None, names=None):
    if names:
        return filter_by_names(data, names)
    if max_k is not None:
        return filter_by_complexity(data, max_k)
    return data


def filter_duplicate_states(data):
    with resources.timing("Removing duplicate states", newline=False):
        _, idxs, inv = np.unique(data.matrix, axis=0, return_index=True, return_inverse=True)
        partition = defaultdict(list)
        for i, rep in enumerate(inv, start=0):
            partition[rep].append(i)
        assert len(partition) == len(idxs) and max(partition.keys()) == len(idxs)-1
        for elems in partition.values():
            hstars = data.labels[elems]
            assert len(hstars) > 0
            for i in range(1, len(hstars)):
                if hstars[i] != hstars[i-1]:
                    print(f"Warning: States {elems[i-1]} and {elems[i]} are indistinguishable but have different hstar values ({hstars[i-1]} vs {hstars[i]})")
                    break

    print(f"Removed {data.numstates()-len(idxs)}/{data.numstates()} duplicate states.")
    return filter_by_row_idx(data, idxs)


def filter_duplicate_features(data):
    raise RuntimeError("To be implemented.")
    # TODO We cannot use np.unique, as we'd like to take feature complexity into account
    #      in order to remove higher-complexity features.
    before = data.numfeatures()
    with resources.timing("Removing duplicate features", newline=False):
        data.matrix = np.unique(data.matrix, axis=1)
    print(f"Removed {before-data.numstates()}/{before} duplicate features.")
    return data


def main(args):
    data = describe(args.matrix)
    data = filter_features(data, max_k=args.k, names=args.include)

    # mi, names_i = remove_linear_combinations(mi, names_i)
    # mi, names_i, complexities_i = extend_feature_pool(mi, names_i, complexities_i, k)

    if args.unique_states:
        data = filter_duplicate_states(data)

    if args.unique_features:
        data = filter_duplicate_features(data)

    if args.rlc:
        data = remove_linear_combinations(data)

    # boolean, numeric = detect_boolean(mi)
    # print(f"Matrix has {len(boolean)} Boolean features + {len(numeric)} numeric features.")

    with resources.timing(f"Fitting {data.matrix.shape} feature matrix to {data.labels.shape} labels", newline=False):
        nonzero_coefs = min(args.m, data.matrix.shape[1])
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=nonzero_coefs)
        omp.fit(data.matrix, data.labels)
    h = print_function(data.names, omp)
    serialize_heuristic(args.heuristic, h)

    score = omp.score(data.matrix, data.labels)
    
    print(f"Predictor score: {score}")
    sample = random.sample(range(data.matrix.shape[0]), 4)
    for i in sample:
        print(f"h*(s_{i}) = {data.labels[i]}, vs h(s_{i}) = {omp.predict(data.matrix[i].reshape(1, -1))}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('matrix', help='The path to the feature matrix')
    parser.add_argument('-m', help='The max. number of non-zero coefficients. Used when running OMP', type=int, default=12)
    parser.add_argument('-k', help='The max. feature complexity to consider', type=int)
    parser.add_argument('--heuristic', help='The file where the resulting heuristic will be serialized', default="heuristic.io")
    parser.add_argument('--include', nargs='*', default=[],
                        help='A list of features, which if present, will be the only ones used')

    parser.add_argument('--rlc', help='Whether to remove linear combinations of features with Sympy', default=False, action='store_true')
    parser.add_argument('--unique_states', help='Remove duplicate states', default=False, action='store_true')
    parser.add_argument('--unique_features', help='Remove duplicate features', default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.matrix):
        raise RuntimeError(f"Could not find matrix file '{args.matrix}'")

    return args


if __name__ == "__main__":
    main(parse_arguments())

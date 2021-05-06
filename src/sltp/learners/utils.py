from collections import defaultdict

import numpy as np
from tarski.utils import resources


class Dataset:
    def __init__(self, names, complexities, matrix, labels):
        self.names = names
        self.complexities = complexities
        self.matrix = matrix
        self.labels = labels

    def complexity_histogram(self):
        histogram = defaultdict(int)
        for c in self. complexities:
            histogram[c] += 1
        return dict(histogram)

    def numfeatures(self):
        return self.matrix.shape[1]

    def numstates(self):
        return self.matrix.shape[0]

    def __str__(self):
        return f"Dataset[{self.matrix.shape}]"
    __repr__ = __str__


def load_compressed(filename):
    data = np.load(filename)
    return Dataset(data['names'], data['complexities'], data['matrix'], data['labels'])


def load(filename):
    with resources.timing("Loading matrices", newline=False):
        loader = load_compressed if filename.endswith('npz') else read
        return loader(filename)


def read(filename):
    with open(filename, "r") as f:
        num_states = sum(1 for _ in f) - 3  # Substract 3 for comment line, feature name and feature cost lines

    with open(filename, "r") as f:
        # Line #0: comment line, simply ignore
        f.readline()

        # Line #1: feature names
        names = np.array(f.readline().rstrip().split(' ')[:-1])

        # Line #2: feature complexities. We remove the complexity corresponding to the label (hstar), which is -1
        complexities = np.array([int(x) for x in f.readline().rstrip().split(' ')][:-1])

        assert len(names) == len(complexities)

        matrix = np.empty(shape=(num_states, len(names)), dtype=np.int8)
        labels = np.empty(shape=(num_states, ), dtype=np.int8)

        # One line per state with the numeric denotation of all features
        for i, line in enumerate(f, start=0):
            row = [int(x) for x in line.rstrip().split(' ')]
            # values.append(row)
            matrix[i] = row[:-1]
            labels[i] = row[-1]

    print(f"Read a matrix with shape {matrix.shape}")
    # print(f"Read a matrix of {len(names)} features per {len(values)} states.")
    # print(f"Value of feature #7 in state #5 is {values[5][7]}")
    # print(f"hstar value of state #5 is {values[5][-1]}")
    return Dataset(names, complexities, matrix, labels)


def detect_boolean(matrix):
    numerics = np.where(matrix.max(0) > 1)[0]
    booleans = [i for i in range(matrix.shape[1]) if i not in numerics]
    return booleans, numerics


def describe(filename):
    data = load(filename)
    boolean, numeric = detect_boolean(data.matrix)
    assert len(boolean) + len(numeric) == data.matrix.shape[1]
    print(f"Matrix has shape {data.matrix.shape}, with {len(boolean)} Boolean and {len(numeric)} numeric features")
    print(f"Complexity histogram: {data.complexity_histogram()}")
    return data


def extend_feature_pool(data, k):
    # poly = PolynomialFeatures(2, interaction_only=True)
    # matrix = poly.fit_transform(matrix)
    # names = poly.get_feature_names(names)
    # return matrix, names

    boolean, numeric = detect_boolean(data.matrix)
    with resources.timing(f"Extending feature pool by conjuncting {len(boolean)} Boolean features with all features",
                          newline=False):
        last_col = data.matrix.shape[1]

        for b in boolean:
            cb = data.complexities[b]
            pair_with = [x for x in range(0, b) if x in numeric] + list(range(b+1, last_col))
            # prefilter those features that will have excessive complexity:
            pair_with = [x for x in pair_with if cb + 1 + data.complexities[x] <= k]
            for x in pair_with:
                result = np.multiply(data.matrix[:, b], data.matrix[:, x])
                data.matrix = np.c_[data.matrix, result]
                data.names = np.r_[data.names, [f"{data.names[b]} {data.names[x]}"]]
                data.complexities = np.r_[data.complexities, [data.complexities[b] + data.complexities[x] + 1]]

    return data

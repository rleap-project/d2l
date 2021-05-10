
#pragma once

#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>

#include <common/utils.h>
#include <blai/matrix.h>
#include <blai/transitions.h>
#include <common/base.h>

namespace sltp {

//! A simple container for a pair of feature matrix and transition sample
class TrainingSet {
public:
    const FeatureMatrix matrix_;
    const TransitionSample transitions_;
    const Sample sample_;

    TrainingSet(FeatureMatrix&& matrix, TransitionSample&& transitions, Sample&& sample) :
        matrix_(std::move(matrix)), transitions_(std::move(transitions)), sample_(std::move(sample))
    {}
    virtual ~TrainingSet() = default;

    const FeatureMatrix& matrix() const { return matrix_; }
    const TransitionSample& transitions() const { return transitions_; }
    const Sample& sample() const { return sample_; }


    friend std::ostream& operator<<(std::ostream &os, const TrainingSet& o) { return o.print(os); }
    std::ostream& print(std::ostream &os) const {

        auto est_size = (double) matrix_.num_features() * matrix_.num_states() * sizeof(FeatureMatrix::feature_value_t) /
                        (1024.0 * 1024.0);

        os
            << "[states: " << transitions_.num_states()
            << ", transitions: " << transitions_.num_transitions()
            << " (" << transitions_.positive().size() << " positive + " << transitions_.negative().size() << " negative)"
            << ", features: " << matrix_.num_features()
            << ", est. size: " << std::setprecision(2) << std::fixed << est_size << " MB.]";
        return os;
    }
};



} // namespaces

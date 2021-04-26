
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

    TrainingSet(FeatureMatrix&& matrix, TransitionSample&& transitions) :
        matrix_(std::move(matrix)), transitions_(std::move(transitions))
    {}
    virtual ~TrainingSet() = default;

    const FeatureMatrix& matrix() const { return matrix_; }
    const TransitionSample& transitions() const { return transitions_; }


    friend std::ostream& operator<<(std::ostream &os, const TrainingSet& o) { return o.print(os); }
    std::ostream& print(std::ostream &os) const {

        auto est_size = (double) matrix_.num_features() * matrix_.num_states() * sizeof(FeatureMatrix::feature_value_t) /
                        (1024.0 * 1024.0);

        unsigned num_alive = 0;
        for (const auto s:transitions_.all_alive()) {
            auto nsuccessors = transitions_.successors(s).size();
            num_alive += nsuccessors;
        }

        os
            << "[states: " << transitions_.num_states()
            << ", transitions: " << transitions_.num_transitions()
            << " (from alive state: " << transitions_.all_alive().size() << ")"
            << ", unsolvable: " << transitions_.num_unsolvable()
            << ", goals: " << transitions_.all_goals().size()
            << ", features: " << matrix_.num_features()
            << ", est. size: " << std::setprecision(2) << std::fixed << est_size << " MB.]";
        return os;
    }
};



} // namespaces


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
    const sltp::Sample sample_;

    TrainingSet(FeatureMatrix&& matrix, TransitionSample&& transitions, sltp::Sample&& sample) :
        matrix_(std::move(matrix)), transitions_(std::move(transitions)), sample_(std::move(sample))
    {}
    virtual ~TrainingSet() = default;

    const FeatureMatrix& matrix() const { return matrix_; }
    const TransitionSample& transitions() const { return transitions_; }
    const sltp::Sample& sample() const { return sample_; }

    bool is_deadend(unsigned s) const {
        return matrix_.is_deadend(s);
    }


    friend std::ostream& operator<<(std::ostream &os, const TrainingSet& o) { return o.print(os); }
    std::ostream& print(std::ostream &os) const {

        auto est_size = (double) matrix_.num_features() * matrix_.num_states() * sizeof(FeatureMatrix::feature_value_t) /
                        (1024.0 * 1024.0);

        unsigned num_alive = 0;
        std::unordered_map<unsigned, unsigned> num_alive_per_instance;

        for (const auto s:transitions_.all_alive()) {
            auto nsuccessors = transitions_.successors(s).size();
            auto instanceid = sample_.state(s).instance_id();
            num_alive += nsuccessors;
            num_alive_per_instance[instanceid] += nsuccessors;
        }

        // here we use the fact that instance IDs are consecutive
        auto ninstances = num_alive_per_instance.size();
        std::string alive_string;
        for (unsigned i=0; i < ninstances; ++i) {
            alive_string += std::to_string(num_alive_per_instance[i]);
            if (i < ninstances-1) alive_string += "/";
        }

        os
            << "[instances: " << ninstances
            << ", states: " << transitions_.num_states()
            << ", transitions: " << transitions_.num_transitions()
            << " (" << num_alive << " alive: " << alive_string << ")"
            << ", deadends: " << matrix_.deadends().size()
            << ", goals: " << matrix_.num_goals()
            << ", features: " << matrix_.num_features()
            << ", est. size: " << std::setprecision(2) << std::fixed << est_size << " MB.]";
        return os;
    }
};



} // namespaces

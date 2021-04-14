
#pragma once

#include <utility>
#include <vector>
#include <cstdint>
#include <string>

#include <blai/matrix.h>
#include <blai/transitions.h>
#include <blai/sample.h>
#include <common/utils.h>
#include <unordered_set>
#include <random>


enum class FeatureValue {
    Eq0 = 0,
    Gt0 = 1,
    Dec = 2,
    Nil = 3,
    Inc = 4
};

inline std::string print_feature_value(const FeatureValue& fval) {
    if (fval == FeatureValue::Eq0) return "=0";
    else if (fval == FeatureValue::Gt0) return ">0";
    else if (fval == FeatureValue::Dec) return "DEC";
    else if (fval == FeatureValue::Nil) return "NIL";
    else if (fval == FeatureValue::Inc) return "INC";
    throw std::runtime_error("Unexpected Feature Value");
}

class DNFPolicy {
public:
    using literal_t = std::pair<uint32_t, FeatureValue>;
    using term_t = std::vector<literal_t>;

    static FeatureValue compute_state_value(unsigned x) { return x>0 ? FeatureValue::Gt0 : FeatureValue::Eq0; }
    static FeatureValue compute_transition_value(unsigned xs, unsigned xsprime) {
        if (xs == xsprime) return FeatureValue::Nil;
        return xs > xsprime ? FeatureValue::Dec : FeatureValue::Inc;
    }

    DNFPolicy() = default;
    explicit DNFPolicy(std::vector<unsigned> features_) :
            features(std::move(features_)), terms()
    {}

    std::vector<unsigned> features;
    std::unordered_set<term_t, sltp::utils::container_hash<term_t>> terms;

};

namespace sltp::cnf {

class StateSpaceSample {
public:
    const FeatureMatrix& matrix_;
    const TransitionSample& transitions_;

    //! The states that belong to this sample
    std::unordered_set<unsigned> states_;

    std::vector<unsigned> alive_states_;
    std::vector<unsigned> goal_states_;
    std::vector<unsigned> nongoal_states_;
    std::vector<unsigned> unknown_states_;
    std::vector<unsigned> expanded_states_;

    StateSpaceSample(const FeatureMatrix& matrix, const TransitionSample& transitions, std::vector<unsigned> states) :
            matrix_(matrix), transitions_(transitions), states_(states.begin(), states.end())
    {
        // Let's classify the states for easier access
        for (unsigned s:states_) {
            if (is_alive(s)) alive_states_.push_back(s);
            else if (is_unknown(s)) unknown_states_.push_back(s);

            // TODO We should find a better way to identify which states have been
            // expanded (e.g. a state could have no successor but have been expanded anyway)
            if (!transitions.successors(s).empty()) expanded_states_.push_back(s);

            if (is_goal(s)) goal_states_.push_back(s);
            else nongoal_states_.push_back(s);
        }
    }

    virtual ~StateSpaceSample() = default;
    StateSpaceSample(const StateSpaceSample&) = default;
//
//    const std::vector<unsigned>& states() const {
//        return states_;
//    }

    inline bool in_sample(unsigned s) const {
        return states_.find(s) != states_.end();
    }

    const FeatureMatrix& matrix() const { return matrix_; }

    //! Return all alive states in this sample
    const std::vector<unsigned>& alive_states() const { return alive_states_; }
    const std::vector<unsigned>& goal_states() const { return goal_states_; }
    const std::vector<unsigned>& nongoal_states() const { return nongoal_states_; }
    const std::vector<unsigned>& unknown_states() const { return unknown_states_; }
    const std::vector<unsigned>& expanded_states() const { return expanded_states_; }

    bool is_goal(unsigned s) const { return transitions_.is_goal(s); }

    bool is_alive(unsigned s) const { return transitions_.is_alive(s); }

    bool is_solvable(unsigned s) const { return is_alive(s) || is_goal(s); }

    bool is_unsolvable(unsigned s) const { return transitions_.is_unsolvable(s); }

    bool is_unknown( unsigned s ) const { return transitions_.is_unknown(s); }

    inline FeatureMatrix::feature_value_t value(unsigned s, unsigned f) const {
        return matrix_.entry(s, f);
    }

    unsigned feature_weight(unsigned f) const {
        return matrix_.feature_cost(f);
    }

    const std::vector<unsigned>& successors(unsigned s) const {
        return transitions_.successors(s);
    }

    const TransitionSample& full_training_set() const {
        return transitions_;
    }

    StateSpaceSample* add_states(const std::vector<unsigned>& states) const {
        std::set<unsigned> tmp(states_.begin(), states_.end());
        tmp.insert(states.begin(), states.end());
        return new StateSpaceSample(matrix_, transitions_, {tmp.begin(), tmp.end()});
    }

    friend std::ostream& operator<<(std::ostream &os, const StateSpaceSample& o) { return o.print(os); }
    std::ostream& print(std::ostream &os) const {
        os << "Sample [sz=" << states_.size() << "]: ";
        for (const auto& s:states_) os << s << ", ";
        os << std::endl;
        return os;
    }
};

void print_classifier(const sltp::FeatureMatrix& matrix, const DNFPolicy& dnf, const std::string& filename);

class StateSampler {
protected:
    std::mt19937& rng;
    const TrainingSet& trset;
    unsigned verbosity;


public:
    StateSampler(std::mt19937& rng, const TrainingSet& trset, unsigned verbosity)
        : rng(rng), trset(trset), verbosity(verbosity)
    {}

    virtual StateSpaceSample* sample_initial_states(unsigned n) = 0;

    virtual std::vector<unsigned> sample_flaws(const DNFPolicy& dnf, unsigned batch_size) = 0;


protected:
    std::vector<unsigned> randomize_all_alive_states(unsigned n = std::numeric_limits<unsigned>::max());
    std::vector<unsigned> sample_flaws(const DNFPolicy& dnf, unsigned batch_size, const std::vector<unsigned>& states_to_check);

};

class RandomSampler : public StateSampler {
public:
    RandomSampler(std::mt19937& rng, const TrainingSet& trset, unsigned verbosity)
        : StateSampler(rng, trset, verbosity)
    {}

    StateSpaceSample* sample_initial_states(unsigned n) override;
    std::vector<unsigned> sample_flaws(const DNFPolicy& dnf, unsigned batch_size) override;

protected:
    using StateSampler::sample_flaws;
};

class GoalDistanceSampler : public StateSampler {
public:
    GoalDistanceSampler(std::mt19937& rng, const TrainingSet& trset, unsigned verbosity)
            : StateSampler(rng, trset, verbosity)
    {}

    StateSpaceSample* sample_initial_states(unsigned n) override;
    std::vector<unsigned> sample_flaws(const DNFPolicy& dnf, unsigned batch_size) override;

    std::vector<unsigned> randomize_and_sort_alive_states(unsigned n = std::numeric_limits<unsigned>::max());

    std::unordered_map<unsigned, unsigned> compute_goal_distance_histogram(const std::vector<unsigned> states);


protected:
    using StateSampler::sample_flaws;
};

class FullSampleSampler : public StateSampler {
public:
    FullSampleSampler(std::mt19937& rng, const TrainingSet& trset, unsigned verbosity)
            : StateSampler(rng, trset, verbosity)
    {}

    StateSpaceSample* sample_initial_states(unsigned n) override;
    std::vector<unsigned> sample_flaws(const DNFPolicy& dnf, unsigned batch_size) override { return {}; }
};


inline std::unique_ptr<StateSampler> select_sampler(const std::string& strategy, std::mt19937& rng, const TrainingSet& trset, unsigned verbosity) {
    if (strategy == "random") return std::make_unique<RandomSampler>(rng, trset, verbosity);
    else if (strategy == "goal") return std::make_unique<GoalDistanceSampler>(rng, trset, verbosity);
    else if (strategy == "full") return std::make_unique<FullSampleSampler>(rng, trset, verbosity);
    else throw std::runtime_error("Unknown state sampling strategy " + strategy);
}


}

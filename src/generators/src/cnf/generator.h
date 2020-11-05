
#pragma once

#include <blai/sample.h>
#include "cnfwriter.h"

//! A feature index
using feature_t = uint32_t;

namespace sltp::cnf {

struct Options {
    enum class Encoding {
        D2L
    };

    //! The path of the workspace where output files will be left
    std::string workspace;

    //! The type of encoding we want to use
    Encoding encoding;

    //! In the d2L encoding, whether we want to exploit the equivalence relation
    //! among transitions given by the feature pool
    bool use_equivalence_classes;

    //! In the d2L encoding, whether we want to exploit the dominance among features to ignore
    //! dominated features and reduce the size of the encoding.
    bool use_feature_dominance;

    //! Whether to be more verbose in the generation of the encoding
    bool verbose;

    //! The slack value for the maximum allowed value for V_\pi(s) = slack * V^*(s)
    double v_slack;

    //! In the d2L encoding, whether to use the incremental refinement approach
    bool use_incremental_refinement;

    //! In the d2L encoding, whether to post constraints to ensure distinguishability of goals
    bool distinguish_goals;

    //! In the d2L encoding, whether to post constraints to ensure distinguishability of goals
    //! and transitions coming from different training instances
    bool cross_instance_constraints;

    //! In the d2L encoding, whether to force any V-descending transition to be labeled as Good
    bool decreasing_transitions_must_be_good;

    //! A list of user-provided feature IDs for which we want to enforce selection
    std::vector<unsigned> enforced_features;

    //! An optional list with the a subset of features (feature IDs) that will be, if present, enforced as Selected;
    //! excluding the rest of features from the pool
    std::vector<unsigned> validate_features;


    [[nodiscard]] bool use_d2l_encoding() const {
        return encoding == Encoding::D2L;
    }
};


enum class CNFGenerationOutput : unsigned {
    Success = 0,
    UnsatTheory = 1,
    ValidationCorrectNoRefinementNecessary = 2
};


inline void undist_goal_warning(unsigned s, unsigned t) {
    std::cout << Utils::warning()
        <<  "No feature can distinguish state " << s << " from state " << t << ", but only one of them is a goal"
        <<  ". The MAXSAT encoding will be UNSAT" << std::endl;
}

inline void undist_deadend_warning(unsigned s, unsigned t) {
    std::cout << Utils::warning()
        <<  "No feature can distinguish state " << s << " from state " << t << ", but (only) one of them is a"
        <<  " dead-end. The MAXSAT encoding will be UNSAT" << std::endl;
}

//! Return a sorted vector with those features that d1-distinguish s from t
std::vector<feature_t> compute_d1_distinguishing_features(const TrainingSet& sample, unsigned s, unsigned t);

//! Return a sorted vector with those features that d2-distinguish transition (s, s') from (t, t')
std::vector<feature_t> compute_d2_distinguishing_features(const TrainingSet& sample,
        unsigned s, unsigned sprime, unsigned t, unsigned tprime);

//! Return a sorted vector with those features that d2-distinguish transition (s, s') from (t, t')
std::vector<feature_t> compute_d1d2_distinguishing_features(
        const std::vector<unsigned>& feature_ids,
        const TrainingSet& sample,
        unsigned s, unsigned sprime, unsigned t, unsigned tprime);

class CNFEncoding {
public:

    CNFEncoding(const TrainingSet& sample, const sltp::cnf::Options& options) :
            tr_set_(sample),
            options(options),
            ns_(sample.matrix().num_states()),
            nf_(sample.matrix().num_features())
    {
        for (unsigned s = 0; s < ns_; ++s) {
            if (is_goal(s)) goals_.push_back(s);
            else nongoals_.push_back(s);
        }
    }

    [[nodiscard]] const std::vector<unsigned>& all_alive() const { return tr_set_.transitions().all_alive(); }

    [[nodiscard]] bool is_goal(unsigned s) const { return tr_set_.matrix().goal(s); }

    [[nodiscard]] bool is_alive(unsigned s) const { return tr_set_.transitions().is_alive(s); }

    [[nodiscard]] bool is_solvable(unsigned s) const { return is_alive(s) || is_goal(s); }

    [[nodiscard]] unsigned feature_weight(unsigned f) const {
        return tr_set_.matrix().feature_cost(f);
    }

    [[nodiscard]] const std::vector<unsigned>& successors(unsigned s) const {
        return tr_set_.transitions().successors(s);
    }

protected:
    //! The transition sample data
    const TrainingSet& tr_set_;

    //! The CNF encoding options
    const sltp::cnf::Options& options;

    //! The number of states in the encoding
    const std::size_t ns_;

    //! The number of features in the encoding
    const std::size_t nf_;

    //! For convenient and performant access, a list of goal and non-goal states
    std::vector<unsigned> goals_, nongoals_;

};


using isomorphism_t = std::unordered_map<unsigned, unsigned>;

//! Check whether all transitions starting in s have some transition starting in t with same qualitative nature
//! on the set of all features in the given feature matrix
bool all_tx_have_analogs(const TrainingSet& sample, unsigned s, unsigned t);

//! Check whether t appears isomorphic to s, and in that case, add it to the given list of isomorphisms
void check_isomorphic(const TrainingSet& sample, unsigned s, unsigned t, isomorphism_t& isomorphisms);

//!
isomorphism_t compute_redundant_states(const TrainingSet& sample);

} // namespaces

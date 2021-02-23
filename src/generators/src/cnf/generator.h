
#pragma once

#include <blai/sample.h>
#include "cnfwriter.h"
#include "options.h"
#include "sample.h"

//! A feature index
using feature_t = uint32_t;

namespace sltp::cnf {

enum class CNFGenerationOutput : unsigned {
    Success = 0,
    UnsatTheory = 1,
    ValidationCorrectNoRefinementNecessary = 2
};


inline void undist_goal_warning(unsigned s, unsigned t) {
    std::cout << sltp::utils::warning()
              <<  "No feature can distinguish state " << s << " from state " << t << ", but only one of them is a goal"
              <<  ". The MAXSAT encoding will be UNSAT" << std::endl;
}

inline void undist_deadend_warning(unsigned s, unsigned t) {
    std::cout << sltp::utils::warning()
              <<  "No feature can distinguish state " << s << " from state " << t << ", but (only) one of them is a"
              <<  " dead-end. The MAXSAT encoding will be UNSAT" << std::endl;
}

//! Return a sorted vector with those features that d1-distinguish s from t
std::vector<feature_t> compute_d1_distinguishing_features(const StateSpaceSample& sample, unsigned s, unsigned t);

//! Return a sorted vector with those features that d2-distinguish transition (s, s') from (t, t')
std::vector<feature_t> compute_d1d2_distinguishing_features(
        const std::vector<unsigned>& feature_ids,
        const StateSpaceSample& sample,
        unsigned s, unsigned sprime, unsigned t, unsigned tprime);


} // namespaces


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
};

//! Return a sorted vector with those features that d1-distinguish s from t
std::vector<feature_t> compute_d1_distinguishing_features(const FeatureMatrix& matrix, unsigned s, unsigned t);

//! Return a sorted vector with those features that d2-distinguish transition (s, s') from (t, t')
std::vector<feature_t> compute_d1d2_distinguishing_features(
        const std::vector<unsigned>& feature_ids,
        const FeatureMatrix& matrix,
        unsigned s, unsigned sprime, unsigned t, unsigned tprime);


} // namespaces

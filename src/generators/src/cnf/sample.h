
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

void print_classifier(const sltp::FeatureMatrix& matrix, const DNFPolicy& dnf, const std::string& filename);

}

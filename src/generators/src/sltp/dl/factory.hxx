
#pragma once

#include <sltp/dl/types.hxx>
#include <common/utils.h>

#include <ctime>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sltp {

struct Predicate;
class Sample;
class TransitionSample;

}

namespace sltp::dl {

class DLBaseElement;
class Concept;
class Role;
class Feature;

class Cache;

//! Command-line option processing
struct Options {
    std::string workspace;
    int timeout;
    unsigned complexity_bound;
    unsigned dist_complexity_bound;
    unsigned cond_complexity_bound;
    bool comparison_features;
    bool print_denotations;
    bool generate_or_concepts;
    bool infty_feat_as_zero;
};

//! We use this to store a number of properties of the denotations of concepts
//! and features over entire samples; properties which we might be interested in analyzing
//! for diverse ends such as pruning redundant features, etc.
struct SampleDenotationProperties {
    bool denotation_is_bool = false;
    bool denotation_is_constant = false;
};

//! A cache from features to their sample denotations
using feature_cache_t = std::unordered_map<
        feature_sample_denotation_t, const Feature*, sltp::utils::container_hash<feature_sample_denotation_t>>;

class Factory {
protected:
    const std::vector<std::string> nominals_;
    std::vector<const Role*> basis_roles_;
    std::vector<const Concept*> basis_concepts_;

    Options options;

    std::vector<const Role*> roles_;

    // A layered set of concepts, concepts_[k] contains all concepts generated in the k-th application of
    // the concept grammar
    std::vector<std::vector<const Concept*> > concepts_;
    std::vector<const Feature*> features_;

    //! Indices of features generated from goal concepts
    std::unordered_set<unsigned> goal_features_;

public:
    Factory(std::vector<std::string>  nominals, Options options) :
            nominals_(std::move(nominals)),
            options(std::move(options))
    {}
    virtual ~Factory() = default;

    inline void insert_basis(const Role *role) {
        basis_roles_.push_back(role);
    }

    inline void insert_basis(const Concept *concept) {
        basis_concepts_.push_back(concept);
    }


    //! Insert the given concept/role as long as it is not redundant with some previous role and it is
    //! below the complexity bound. Return whether the role was effectively inserted.
    template <typename T1, typename T2>
    bool attempt_insertion(const T1& elem, Cache &cache, const Sample &sample, std::vector<const T2*>& container) const;

    bool check_timeout(const std::clock_t& start_time) const {
        if (options.timeout <= 0) return false;
        double elapsed = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
        if (elapsed > options.timeout) {
            std::cout << "\tTimeout of " << options.timeout << " sec. reached while generating concepts" << std::endl;
            return true;
        }
        return false;
    }

    // apply one iteration of the concept generation grammar
    // new concepts are left on a new last layer in concepts_
    // new concepts are non-redundant if sample != nullptr
    std::pair<int, int> advance_step(unsigned target_k, Cache &cache, const Sample &sample, const std::clock_t& start_time, const std::vector<std::vector<const Role*>>& roles_by_complexity);

    //! Retrieve the predicate at the basis of a given role (given the current grammar restrictions, there will be
    //! exactly one such predicate
    static const Predicate* get_role_predicate(const Role* r);

    void generate_basis(const Sample &sample);

    std::vector<const Concept*> generate_goal_concepts_and_roles(Cache &cache, const Sample &sample);

    int generate_roles(Cache &cache, const Sample &sample);

    std::vector<const Concept*> generate_concepts(Cache &cache, const Sample &sample, const std::clock_t& start_time);

    void generate_comparison_features(
            Cache& cache,
            const Sample& sample,
            const TransitionSample& transitions,
            feature_cache_t& seen_denotations);

    void generate_conditional_features(
            Cache& cache,
            const Sample& sample,
            const TransitionSample& transitions,
            feature_cache_t& seen_denotations);

    void generate_features(
            const std::vector<const Concept*>& concepts,
            Cache &cache, const Sample &sample,
            const TransitionSample& transitions,
            const std::vector<const Concept*>& forced_goal_features);

    void print_feature_count() const;

    bool attempt_cardinality_feature_insertion(
            const Concept* c,
            Cache &cache,
            const Sample &sample,
            const TransitionSample& transitions,
            feature_cache_t& seen_denotations,
            bool check_redundancy);

    //! Insert the given feature if its complexity is below the given bound, its denotation is not constant,
    //! and its denotation trail is not redundant with that of some previously-generated feature
    //! Return whether the feature was indeed inserted or not
    bool attempt_feature_insertion(
            const Feature* feature, unsigned bound,
            Cache &cache, const Sample &sample,
            const TransitionSample& transitions, feature_cache_t& seen_denotations,
            bool check_redundancy);

    void generate_distance_features(
            const std::vector<const Concept*>& concepts, Cache &cache,
            const Sample &sample,
            const TransitionSample& transitions,
            feature_cache_t& seen_denotations);

    static bool prune_feature_denotation(
            const Feature& f,
            const feature_sample_denotation_t& fd,
            const SampleDenotationProperties& properties,
            const Sample &sample,
            const TransitionSample& transitions,
            feature_cache_t& seen_denotations,
            bool check_redundancy);

    std::ostream& report_dl_data(std::ostream &os) const;

    void output_feature_matrix(std::ostream &os, const Cache &cache, const Sample &sample, const sltp::TransitionSample& transitions) const;

    void log_all_concepts_and_features(const std::vector<const Concept*>& concepts,
                                       const Cache &cache, const Sample &sample,
                                       const std::string& workspace, bool print_denotations);

    //! Return all generated concepts in a single, *unlayered* vector, and sorted by increasing complexity
    std::vector<const Concept*> consolidate_concepts() const;

    int generate_goal_equality_concepts(const Sample& sample, Cache& cache);
};

} // namespaces

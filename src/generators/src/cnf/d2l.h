
#pragma once

#include <common/helpers.h>
#include <cnf/sample.h>
#include <cnf/generator.h>
#include <cnf/types.h>
#include <cnf/solver.h>

#include <numeric>


namespace sltp::cnf {

//! A map between feature IDs and transition IDs) and
//! the CNF variable IDs that tell us whether features are selected in the solution and transitions are labeled as
//! good in the solution
struct VariableMapping {
    //! A map from each feature index to the SAT variable ID of Selected(f)
    std::vector<cnfvar_t> selecteds;

    explicit VariableMapping(unsigned nfeatures) : selecteds(nfeatures, std::numeric_limits<uint32_t>::max())
    {}
};

class D2LEncoding {
public:
    D2LEncoding(const TrainingSet& sample, const Options& options) :
            sample_(sample),
            options(options),
            nf_(sample.matrix().num_features()),
            feature_ids(),
            positive(),
            negative()
    {
        if (!options.validate_features.empty()) {
            // Consider only the features we want to validate
            feature_ids = options.validate_features;
        } else { // Else, we will simply consider all feature IDs
            feature_ids.resize(nf_);
            std::iota(feature_ids.begin(), feature_ids.end(), 0);
        }

        compute_equivalence_relations();
    }

    virtual ~D2LEncoding() = default;

    virtual std::pair<cnf::CNFGenerationOutput, VariableMapping> generate(CNFWriter& wr);

    inline unsigned get_transition_id(state_id_t s, state_id_t t) const { return sample_.transitions().get_transition_id(s, t); }
    inline const state_pair& get_state_pair(unsigned tx) const { return sample_.transitions().get_state_pair(tx); }

    //! Whether the two given transitions are distinguishable through the given features alone
    bool are_transitions_d1d2_distinguishable(
            state_id_t s, state_id_t sprime, state_id_t t, state_id_t tprime, const std::vector<unsigned>& features) const;

    DNFPolicy generate_dnf_from_solution(const VariableMapping& variables, const SatSolution& solution) const;

    DNFPolicy generate_dnf(const std::vector<unsigned>& selecteds) const;

protected:
    //! The transition sample data
    const TrainingSet& sample_;

    //! The CNF encoding options
    const Options& options;

    //! The number of features in the encoding
    const std::size_t nf_;

    //! The only feature IDs that we will consider for the encoding
    std::vector<unsigned> feature_ids;

    std::unordered_set<unsigned> positive;
    std::unordered_set<unsigned> negative;

    //!
    void compute_equivalence_relations();

    int process_transition(unsigned txid, std::unordered_map<transition_trace, unsigned>& from_trace_to_class_repr);
};

} // namespaces


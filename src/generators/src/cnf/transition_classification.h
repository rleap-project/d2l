
#pragma once

#include <common/helpers.h>
#include "generator.h"
#include "types.h"

#include <numeric>


namespace sltp::cnf {

class D2LEncoding : public CNFEncoding {
public:
    enum class transition_type : bool {
        alive_to_solvable,
        alive_to_dead
    };

    D2LEncoding(const TrainingSet& sample, const Options& options) :
            CNFEncoding(sample, options),
            transition_ids_(),
            transition_ids_inv_(),
            class_representatives_(),
            from_transition_to_eq_class_(),
            types_(),
            necessarily_bad_transitions_(),
            feature_ids()
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


    sltp::cnf::CNFGenerationOutput write(CNFWriter& wr, const std::vector<transition_pair>& transitions_to_distinguish);

    inline unsigned get_transition_id(state_id_t s, state_id_t t) const { return transition_ids_.at(state_pair(s, t)); }

    inline unsigned get_representative_id(unsigned tx) const { return from_transition_to_eq_class_.at(tx); }

    inline unsigned get_class_representative(state_id_t s, state_id_t t) const {
        return get_representative_id(get_transition_id(s, t));
    }

    inline const state_pair& get_state_pair(unsigned tx) const { return transition_ids_inv_.at(tx); }

    inline bool is_necessarily_bad(unsigned tx) const {
        return necessarily_bad_transitions_.find(tx) != necessarily_bad_transitions_.end();
    }

    inline int get_vstar(unsigned s) const {
        int vstar = tr_set_.transitions().vstar(s);
        return vstar < 0 ? -1 : vstar;
    }

    inline int get_max_v(unsigned s) const {
        int vstar = tr_set_.transitions().vstar(s);
        return vstar < 0 ? -1 : std::ceil(options.v_slack * vstar);
    }

    inline unsigned compute_D() const {
        // return 20;
        // D will be the maximum over the set of alive states of the upper bounds on V_pi
        unsigned D = 0;
        for (const auto s:all_alive()) {
            auto max_v_s = get_max_v(s);
            if (max_v_s > D) D = max_v_s;
        }
        return D;
    }

    using flaw_index_t = std::unordered_map<transition_id_t, std::vector<transition_id_t>>;
    bool check_existing_solution_for_flaws(flaw_index_t& flaws) const;

    //! Whether the two given transitions are distinguishable through the given features alone
    bool are_transitions_d1d2_distinguishable(
            state_id_t s, state_id_t sprime, state_id_t t, state_id_t tprime, const std::vector<unsigned>& features) const;

    CNFGenerationOutput refine_theory(CNFWriter& wr);

protected:
    // A mapping from pairs of state to the assigned transition id
    std::unordered_map<state_pair, unsigned, boost::hash<state_pair>> transition_ids_;
    // The reverse mapping
    std::vector<state_pair> transition_ids_inv_;

    // A list of transition IDs that are the representative of their class
    std::vector<unsigned> class_representatives_;

    // A mapping from the ID of the transition to the ID of its equivalence class
    std::vector<unsigned> from_transition_to_eq_class_;

    // A mapping from the ID of the transition to its transition type
    std::vector<transition_type> types_;

    std::unordered_set<unsigned> necessarily_bad_transitions_;

    //! The only feature IDs that we will consider for the encoding
    std::vector<unsigned> feature_ids;

    //!
    void compute_equivalence_relations();

    //! Return a list with the IDs of those features that are not dominated by other features
    std::vector<unsigned> prune_dominated_features();


    //! Return DT(f), the set of pairs of transitions that are distinguished by the given feature f
    std::vector<transition_pair> compute_dt(unsigned f);

    std::vector<transition_pair> compute_transitions_to_distinguish(
            bool load_transitions_from_previous_iteration, const flaw_index_t& flaws) const;

    std::vector<transition_pair> distinguish_all_transitions() const;

    void store_transitions_to_distinguish(const std::vector<transition_pair> &transitions) const;

    std::vector<transition_pair> load_transitions_to_distinguish() const;

    std::vector<transition_pair> generate_t0_transitions(unsigned m=40) const;

    void report_eq_classes() const;
};

} // namespaces


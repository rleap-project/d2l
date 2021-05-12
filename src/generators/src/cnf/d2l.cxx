
#include "d2l.h"
#include "types.h"

#include <iostream>
#include <vector>
#include <unordered_map>

#include <boost/functional/hash.hpp>
#include <common/helpers.h>


namespace sltp::cnf {


transition_denotation compute_transition_denotation(FeatureMatrix::feature_value_t s_f, FeatureMatrix::feature_value_t sprime_f) {
    int type_s = (int) sprime_f - (int) s_f; // <0 if DEC, =0 if unaffected, >0 if INC
    int sign = (type_s > 0) ? 1 : ((type_s < 0) ? -1 : 0); // Get the sign
    return transition_denotation(bool(s_f > 0), sign);
}


//! Return an integer with the ID of the redundant transition, if any, or -1, if none (i.e. if the given transition
//! is actually distinguishable from all previously seen transitions)
int D2LEncoding::process_transition(
        unsigned txid, std::unordered_map<transition_trace, unsigned>& from_trace_to_class_repr) {
    const auto& [s, sprime] = get_state_pair(txid);

    // Compute the trace of the transition for all features
    transition_trace trace;
    for (auto f:feature_ids) {
        trace.denotations.emplace_back(compute_transition_denotation(sample_.matrix().entry(s, f), sample_.matrix().entry(sprime, f)));
    }

    // TODO Clean this up
    // Check whether some previous transition has the same transition trace
    auto it = from_trace_to_class_repr.find(trace);
    if (it == from_trace_to_class_repr.end()) {
        // We have a new equivalence class, to which we assign the ID of the representative transition
//        from_transition_to_eq_class_.push_back(txid);
        from_trace_to_class_repr.emplace(trace, txid);
//        class_representatives_.push_back(txid);
        return -1;

    } else {
        // We already have other transitions undistinguishable from this one
//        assert(it->second < id);
//        from_transition_to_eq_class_.push_back(it->second);
        return (int) it->second;
    }
}

void D2LEncoding::compute_equivalence_relations() {
    if (!options.use_equivalence_classes) {
        // If we don't want to compute equivalences, simply take into consideration all positive and negative examples.
        positive = std::unordered_set<unsigned>(sample_.transitions().positive());
        negative = std::unordered_set<unsigned>(sample_.transitions().negative());
        return;
    }

    std::cout << "Computing indistinguishability for " << sample_.transitions().positive().size() << " positive and "
              << sample_.transitions().negative().size() << " negative examples" << std::endl;

    // Note that the code above relies for correctness on the fact that negative examples are dealt with before
    // positive examples. This way, we can determine whether positive examples are redundant with respect to
    // a negative or a positive example and report this to the user accurately.
    unsigned negative_redundant = 0, positive_undistinguishable = 0, positive_redundant = 0;
    // A mapping from a full transition trace to the ID of the corresponding equivalence class
    std::unordered_map<transition_trace, unsigned> from_trace_to_class_repr;
    for (const auto& txid:sample_.transitions().negative()) {
        auto redundant_id = process_transition(txid, from_trace_to_class_repr);
        if (redundant_id==-1) negative.insert(txid);
        else ++negative_redundant;
    }

    for (const auto& txid:sample_.transitions().positive()) {
        auto redundant_id = process_transition(txid, from_trace_to_class_repr);
        if (redundant_id==-1) positive.insert(txid);
        else if (negative.count(redundant_id)>0) ++positive_undistinguishable;
        else ++positive_redundant;
    }

    // Print some stats
        if (options.verbosity > 0) {
        std::cout << "Number of redundant negative examples: " << negative_redundant << std::endl;
        std::cout << "Number of redundant positive examples: " << positive_redundant << std::endl;
        std::cout << "Number of positive examples that are undist. from a negative example: "
                  << positive_undistinguishable << std::endl;
    }
}


std::pair<cnf::CNFGenerationOutput, VariableMapping> D2LEncoding::generate(CNFWriter& wr)
{
    using Wr = CNFWriter;
    const auto& mat = sample_.matrix();

    VariableMapping variables(nf_);

    auto varmapstream = utils::get_ofstream(options.workspace + "/varmap.wsat");

    unsigned n_selected_clauses = 0;
    unsigned n_separation_clauses = 0;


    if (options.verbosity>0) {
        std::cout << "Generating CNF encoding to distinguish " << positive.size() << " positive transitions from "
                  << negative.size() << " negative transitions." << std::endl;
    }

    /////// CNF variables ///////
    // Create one "Select(f)" variable for each feature f in the pool
    for (unsigned f:feature_ids) {
        auto v = wr.var("Select(" + mat.feature_name(f) + ")");
        variables.selecteds[f] = v;
    }

    // From this point on, no more variables will be created. Print total count.
    if (options.verbosity>0) {
        std::cout << "A total of " << wr.nvars() << " Select(f) variables were created" << std::endl;
//        std::cout << "\tSelect(f): " << variables.selecteds.size() << std::endl;
    }
    // Check our variable count is correct
    assert(wr.nvars() == variables.selecteds.size() );

    /////// CNF constraints ///////
    auto ninstances = sample_.sample().num_instances();

//    std::mt19937 rng(options.seed);
//    std::uniform_real_distribution<> dist(0, 1);

    // Let's group positive and negative transitions by the instance they belong to
    std::vector<std::vector<unsigned>> pos_per_instance(ninstances);
    std::vector<std::vector<unsigned>> neg_per_instance(ninstances);
    for (const auto tid:positive) {
        const auto& [s, sprime] = get_state_pair(tid);
        assert(sample_.sample().instance_id(s) == sample_.sample().instance_id(sprime));
        pos_per_instance.at(sample_.sample().instance_id(s)).push_back(tid);
    }
    for (const auto tid:negative) {
        const auto& [s, sprime] = get_state_pair(tid);
        assert(sample_.sample().instance_id(s) == sample_.sample().instance_id(sprime));
        neg_per_instance.at(sample_.sample().instance_id(s)).push_back(tid);
    }


    // Clauses (6), (7):
    for (unsigned iid=0; iid < ninstances; ++iid) {
//    for (unsigned iid=0; iid < 1; ++iid) {
        for (const auto pos_id:pos_per_instance.at(iid)) {
//        for (const auto pos_id:positive) {
            const auto&[s, sprime] = get_state_pair(pos_id);

            for (const auto neg_id:neg_per_instance.at(iid)) {
//                if (dist(rng) < 0.6) continue;

//            for (const auto neg_id:negative) {
                const auto&[t, tprime] = get_state_pair(neg_id);
                cnfclause_t clause;

                // Compute first the Selected(f) terms
                const auto dist = compute_d1d2_distinguishing_features(feature_ids, sample_.matrix(), s, sprime, t,
                                                                       tprime);
                if (dist.empty()) {
                    std::cout << sltp::utils::warning()
                              << "No feature distinguishes positive transition (" << s << ", " << sprime
                              << ") from negative transition (" << t << ", " << tprime << "). Theory is UNSAT"
                              << std::endl;
                }

                for (feature_t f:dist) {
                    clause.push_back(Wr::lit(variables.selecteds.at(f), true));
                }

                wr.cl(clause);
                n_separation_clauses += 1;
            }
        }
    }

//        std::cout << "Posting (weighted) soft constraints for " << variables.selecteds.size() << " features" << std::endl;
    for (unsigned f:feature_ids) {
        if (sample_.matrix().feature_cost(f) == 0) {
            std::cout << sample_.matrix().feature_name(f) << std::endl;
            std::cout << "Done" << std::endl;
        }
        wr.cl({Wr::lit(variables.selecteds[f], false)}, sample_.matrix().feature_cost(f));
    }

    n_selected_clauses += feature_ids.size();

    if (options.verbosity>0) {
        // Print a breakdown of the clauses
        std::cout << "A total of " << wr.nclauses() << " clauses were created" << std::endl;
        std::cout << "\tTransition separation: " << n_separation_clauses << std::endl;
        std::cout << "\t(Weighted) Select(f): " << n_selected_clauses << std::endl;
        assert(wr.nclauses() == n_selected_clauses + n_separation_clauses);
    }
    return {CNFGenerationOutput::Success, variables};
}


DNFPolicy D2LEncoding::generate_dnf(const std::vector<unsigned>& selecteds) const {
    DNFPolicy dnf(selecteds);
    for (const auto& tid:positive) {
        const auto& [s, sprime] = get_state_pair(tid);

        DNFPolicy::term_t clause;

        for (const auto& f:selecteds) {
            const auto& fs = sample_.matrix().entry(s, f);
            const auto& fsprime = sample_.matrix().entry(sprime, f);

            clause.emplace_back(f, DNFPolicy::compute_state_value(fs));
            clause.emplace_back(f, DNFPolicy::compute_transition_value(fs, fsprime));
        }

        dnf.terms.insert(clause);
    }
    return dnf;
}


DNFPolicy D2LEncoding::generate_dnf_from_solution(const VariableMapping& variables, const SatSolution& solution) const {
    // Let's parse the relevant bits of the CNF solution:
    std::vector<unsigned> selecteds;
    for (unsigned f=0; f < variables.selecteds.size(); ++f) {
        auto varid = variables.selecteds[f];
        if (varid>0 && solution.assignment.at(varid)) selecteds.push_back(f);
    }
    return generate_dnf(selecteds);
}

bool D2LEncoding::are_transitions_d1d2_distinguishable(
        state_id_t s, state_id_t sprime, state_id_t t, state_id_t tprime, const std::vector<unsigned>& features) const {
    const auto& mat = sample_.matrix();
    for (unsigned f:features) {
        if (are_transitions_d1d2_distinguished(mat.entry(s, f), mat.entry(sprime, f),
                                               mat.entry(t, f), mat.entry(tprime, f))) {
            return true;
        }
    }
    return false;
}

} // namespaces
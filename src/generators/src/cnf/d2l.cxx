
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


void D2LEncoding::compute_equivalence_relations() {
    // A mapping from a full transition trace to the ID of the corresponding equivalence class
    std::unordered_map<transition_trace, unsigned> from_trace_to_class_repr;

    for (const auto s:sample_.expanded_states()) {
        for (unsigned sprime:sample_.successors(s)) {
            auto tx = std::make_pair((state_id_t) s, (state_id_t) sprime);
            auto id = (unsigned) transition_ids_inv_.size(); // Assign a sequential ID to the transition

            transition_ids_inv_.push_back(tx);
            auto it1 = transition_ids_.emplace(tx, id);
            assert(it1.second);

            if (!sample_.is_solvable(sprime)) { // An alive-to-dead transition cannot be Good
                necessarily_bad_transitions_.emplace(id);
            }

            if (!options.use_equivalence_classes) {
                // If we don't want to use equivalence classes, we simply create one fictitious equivalence class
                // for each transition, and proceed as usual
                from_transition_to_eq_class_.push_back(id);
                class_representatives_.push_back(id);
                continue;
            }

            // Compute the trace of the transition for all features
            transition_trace trace;
            for (auto f:feature_ids) {
                trace.denotations.emplace_back(
                        compute_transition_denotation(sample_.value(s, f), sample_.value(sprime, f)));
            }

            // Check whether some previous transition has the same transition trace
            auto it = from_trace_to_class_repr.find(trace);
            if (it == from_trace_to_class_repr.end()) {
                // We have a new equivalence class, to which we assign the ID of the representative transition
                from_transition_to_eq_class_.push_back(id);
                from_trace_to_class_repr.emplace(trace, id);
                class_representatives_.push_back(id);
            } else {
                // We already have other transitions undistinguishable from this one
                assert(it->second < id);
                from_transition_to_eq_class_.push_back(it->second);

//                if (types_[it->second] != types_[id]) {
//                    // We have to non-distinguishable transitions, one from alive to solvable, and one from alive
//                    // to dead; hence, both will need to be labeled as not Good
//                    throw std::runtime_error("Found two non-distinguishable transitions with different types");
//                }
            }
        }
    }

    // All transitions that belong to some class where at least one transition must be bad, must be bad
    std::unordered_set<unsigned> necessarily_bad_classes;
    for (const auto id:necessarily_bad_transitions_) {
        necessarily_bad_classes.insert(get_representative_id(id));
    }

    for (unsigned id=0; id < transition_ids_.size(); ++id) {
        auto repr = get_representative_id(id);
        if (necessarily_bad_classes.find(repr) != necessarily_bad_classes.end()) {
            necessarily_bad_transitions_.insert(id);
        }
    }

    // Print some stats
    if (options.verbosity > 0) {
        std::cout << "Number of transitions/equivalence classes: " << transition_ids_.size()
                  << "/" << class_representatives_.size() << std::endl;
        std::cout << "Number of necessarily bad transitions/classes: " << necessarily_bad_transitions_.size()
                  << "/" << necessarily_bad_classes.size() << std::endl;
    }

//    report_eq_classes();
}

void D2LEncoding::report_eq_classes() const {
    std::unordered_map<unsigned, std::vector<state_pair>> classes;
    for (unsigned tx=0; tx < transition_ids_.size(); ++tx) {
        auto repr = get_representative_id(tx);
        classes[repr].push_back(get_state_pair(tx));
    }

    unsigned i = 0;
    for (const auto& elem:classes) {
        std::cout << "Class " << ++i << ": " << std::endl;
        const auto& elements = elem.second;
        for (unsigned j = 0; j < elements.size(); ++j) {
            const auto& txpair = elements[j];
            std::cout << "(" << txpair.first << ", " << txpair.second << ")";
            if (j < elements.size()-1) std::cout << "; ";
        }
        std::cout << std::endl << std::endl;
    }
}


CNFGenerationOutput D2LEncoding::generate_asp_instance_1(std::ofstream& os) {
    const auto& mat = sample_.matrix();
    const unsigned max_d = compute_D();

    os << "#const maxd=" << max_d << "." << std::endl;
    os << "#const delta=" << options.v_slack << "." << std::endl;
    os << std::endl;

    os << "state(0.." << mat.num_states() << ")." << std::endl;
    os << "feature(0.." << mat.num_features() << ")." << std::endl;
    os << std::endl;

    for (unsigned s = 0; s < mat.num_states(); ++s) {
        if (!sample_.is_solvable(s)) {
            os << "dead(" << s << ")." << std::endl;
        }
        else if (sample_.is_goal(s)) {
            os << "goal(" << s << ")." << std::endl;
        }
        else {
            os << "alive(" << s << ")." << std::endl;
            os << "vstar(" << s << ", " << get_vstar(s) << ")." << std::endl;
        }

        for (unsigned sprime:sample_.successors(s)) {
            os << "transition(" << s << ", " << sprime << ")." << std::endl;
        }
    }
    os << std::endl;

    for (unsigned f = 0; f < mat.num_features(); ++f) {
        for (unsigned s = 0; s < mat.num_states(); ++s) {
            os << "val(" << f << ", " << s << ", " << mat.entry(s, f) << ")." << std::endl;
        }
    }
    os << std::endl;


    for (unsigned f = 0; f < mat.num_features(); ++f) {
        os << "weight(" << f << ", " << sample_.feature_weight(f) << ")." << std::endl;
    }
    os << std::endl;

    return CNFGenerationOutput::Success;
}


CNFGenerationOutput D2LEncoding::generate_asp_instance_10(std::ofstream& os) {
    unsigned n_bad_transitions = 0;
    unsigned n_separation_clauses = 0;
    unsigned n_goal_clauses = 0;
    unsigned nrules = 0;

    const auto& mat = sample_.matrix();

//    os << "state(0.." << mat.num_states() << ")." << std::endl;
    os << "%% We have a total of " << mat.num_features() << " features" << std::endl;
    os << "feature(0.." << mat.num_features()-1 << ")." << std::endl;
    os << std::endl;

    os << "%% There are " << sample_.states_.size() << " in the sample" << std::endl;
    for (auto s:sample_.states_) os << "sampled(" << s << "). ";
    os << std::endl << std::endl;

    os << "%% Sample description" << std::endl;
    for (auto s = 0; s < mat.num_states(); ++s) {
        if (!sample_.in_sample(s)) continue;

        if (!sample_.is_solvable(s)) {
            os << "dead(" << s << ")." << std::endl;
        }
        else if (sample_.is_goal(s)) {
            os << "goal(" << s << ")." << std::endl;
        }
        else {
            os << "alive(" << s << ")." << std::endl;

            for (unsigned sprime:sample_.successors(s)) {
                os << "transition(" << s << ", " << sprime << ")." << std::endl;
                const auto& [t, tprime] = get_state_pair(get_class_representative(s, sprime));
                os << "repr(" << s << ", " << sprime << ", " << t << ", " << tprime << " )." << std::endl;
                nrules += 1;
            }
        }
        nrules += 1;
    }
    os << std::endl;


    if (options.distinguish_goals) {
        os << "%% Goal distinguishability (" << sample_.goal_states().size() << " goals):" << std::endl;
        for (unsigned s:sample_.goal_states()) {
            for (unsigned t:sample_.nongoal_states()) {

                const auto d1feats = compute_d1_distinguishing_features(sample_, s, t);
                if (d1feats.empty()) {
                    undist_goal_warning(s, t);
                    return CNFGenerationOutput::UnsatTheory;
                }

                os << ":-";
                for (unsigned i=0, m=d1feats.size(); i<m; ++i) {
                    os << "not sel(" << d1feats[i];
                    if (i < m-1) os << "), ";
                }
                os << ")." << std::endl;
                n_goal_clauses += 1;
                nrules += 1;
            }
        }
        os << std::endl;
    }

    os << "%% Bad transitions: " << std::endl;
    for (const auto tx1:class_representatives_) {
        if (is_necessarily_bad(tx1)) {
            const auto& [s, sprime] = get_state_pair(tx1);
            os << ":- good(" << s << ", " << sprime << ")." << std::endl;
            n_bad_transitions += 1;
            nrules += 1;
        }
    }
    os << std::endl;


    auto transitions_to_distinguish = distinguish_all_transitions();
    os << "%% Distinguishability constraints for " << transitions_to_distinguish.size() << " pairs of transitions" << std::endl;
    for (const auto& tpair:transitions_to_distinguish) {
        assert (!is_necessarily_bad(tpair.tx1));
        const auto& [s, sprime] = get_state_pair(tpair.tx1);
        const auto& [t, tprime] = get_state_pair(tpair.tx2);

        os << ":- good(" << s << ", " << sprime << "), not good(" << t << ", " << tprime << ")";
        for (feature_t f:compute_d1d2_distinguishing_features(feature_ids, sample_, s, sprime, t, tprime)) {
            os << ", " << "not sel(" << f << ")";
        }
        os << "." << std::endl;
        n_separation_clauses += 1;
        nrules += 1;
    }
    os << std::endl;

    os << "%% Feature weights" << std::endl;
    for (unsigned f = 0; f < mat.num_features(); ++f) {
        os << "weight(" << f << ", " << sample_.feature_weight(f) << ")." << std::endl;
        nrules += 1;
    }
    os << std::endl;

    // Print a breakdown of the clauses
    std::cout << "A total of " << nrules << " ground rules were created" << std::endl;

    return CNFGenerationOutput::Success;
}


std::pair<cnf::CNFGenerationOutput, VariableMapping> D2LEncoding::generate(CNFWriter& wr)
{
    using Wr = CNFWriter;
    const auto& mat = sample_.matrix();
    const auto num_transitions = transition_ids_.size();

    VariableMapping variables(nf_);

    auto varmapstream = utils::get_ofstream(options.workspace + "/varmap.wsat");

    // A map from pairs of states (s, s') to the ID of the SAT variable reach(s, s')
    std::unordered_map<state_pair, cnfvar_t, boost::hash<state_pair>> reach;

    // Map from state and card. const. var pairs (s,y) to SAT variable ID of the variable CardConstraint(s,y)
    std::unordered_map<std::pair<unsigned,unsigned>, cnfvar_t, boost::hash<state_pair> > card_constraint_vars;

    unsigned n_descending_clauses = 0;
    unsigned n_v_function_clauses = 0;
    unsigned n_good_tx_clauses = 0;
    unsigned n_selected_clauses = 0;
    unsigned n_separation_clauses = 0;
    unsigned n_goal_clauses = 0;
    unsigned n_card_constraints = 0;
    bool with_card_constraints = false;


    const unsigned max_d = compute_D();
    if (options.verbosity>0) {
        std::cout << "Generating CNF encoding for " << sample_.expanded_states().size() << " expanded states, "
                  <<  transition_ids_.size() << " alive-to-solvable and alive-to-dead transitions and "
                  << class_representatives_.size() << " transition equivalence classes with a d_max=" << max_d << std::endl;
    }

    /////// CNF variables ///////
    // Create one "Select(f)" variable for each feature f in the pool
    for (unsigned f:feature_ids) {
        auto v = wr.var("Select(" + mat.feature_name(f) + ")");
        variables.selecteds[f] = v;
    }

    if( with_card_constraints ) {
        unsigned aux = max_d;
        while (aux > 0) {
            n_card_constraints++;
            aux >>= 1;
        }
        for (const auto s:sample_.alive_states()) {
            for (unsigned y = 0; y < n_card_constraints; y++) {
                auto var = wr.var("CardConstraint(" + std::to_string(s) + "," + std::to_string(y) + ")");
                card_constraint_vars.emplace(std::make_pair(s, y), var);
            }
        }
    }

    // Create variables V(s, d) variables for all alive state s and d in 1..D
    for (const auto s:sample_.expanded_states()) {
        const auto min_vs = get_vstar(s);
        const auto max_vs = get_max_v(s);
//        std::cout << min_vs << ", " << max_vs << std::endl;
//        assert(max_vs > 0 && max_vs <= max_d && min_vs >= 0 && min_vs <= max_vs);

        cnfclause_t within_range_clause;

        // TODO Note that we could be more efficient here and create only variables V(s,d) for those values of d that
        //  are within the bounds below. I'm leaving that as a future optimization, as it slightly complicates the
        //  formulation of constraints C2
//        std::cout << "V(" << s << "): [" << min_vs << ", " << max_vs << "]" << std::endl;
        for (unsigned d = 1; d <= max_d; ++d) {
            const auto v_s_d = wr.var("V(" + std::to_string(s) + ", " + std::to_string(d) + ")");
            variables.vs.emplace(std::make_pair(s, d), v_s_d);
//                std::cout << s << ", " << d << std::endl;

            if (min_vs < 0 || (d >= min_vs && d <= max_vs)) {
                within_range_clause.push_back(Wr::lit(v_s_d, true));
            }
        }

        // Add clauses (4), (5)
        wr.cl(within_range_clause);
        n_v_function_clauses += 1;

        for (unsigned d = 1; d <= max_d; ++d) {
            if( with_card_constraints ) {
                for( unsigned y = 0; y < n_card_constraints; y++ ) {
                    bool eval = ( ( (d-1) & (1 << y ) ) > 0 );
                    wr.cl({
                                  wr.lit( variables.vs.at({s,d}), false ),
                                  wr.lit( card_constraint_vars.at({s,y}), eval )
                          });
                    n_v_function_clauses += 1;
                }
            }
            else {
                for (unsigned dprime = d+1; dprime <= max_d; ++dprime) {
                    wr.cl({Wr::lit(variables.vs.at({s, d}), false), Wr::lit(variables.vs.at({s, dprime}), false)});
                    n_v_function_clauses += 1;
                }
            }
        }
    }


    // Create a variable "Good(s, s')" for each transition (s, s') such that s' is solvable and (s, s') is not in BAD
    for (unsigned tx=0; tx < num_transitions; ++tx) {
        if (is_necessarily_bad(tx)) continue; // This includes  alive-to-dead transitions

        const auto& txpair = get_state_pair(tx);
        const auto s = txpair.first;
        const auto sprime = txpair.second;

        cnfvar_t good_s_sprime = 0;
        auto repr = get_representative_id(tx);
        if (tx == repr) { // tx is an equivalence class representative: create the Good(s, s') variable
            good_s_sprime = wr.var("Good(" + std::to_string(s) + ", " + std::to_string(sprime) + ")");
            auto it = variables.goods.emplace(tx, good_s_sprime);
            assert(it.second); // i.e. the SAT variable Good(s, s') is necessarily new
            varmapstream << good_s_sprime << " " << s << " " << sprime << std::endl;

        } else {  // tx is represented by repr, no need to create a redundant variable
            good_s_sprime = variables.goods.at(repr);
        }
    }


    // From this point on, no more variables will be created. Print total count.
    if (options.verbosity>0) {
        std::cout << "A total of " << wr.nvars() << " variables were created" << std::endl;
        std::cout << "\tSelect(f): " << variables.selecteds.size() << std::endl;
        std::cout << "\tGood(s, s'): " << variables.goods.size() << std::endl;
        std::cout << "\tV(s, d): " << variables.vs.size() << std::endl;
        std::cout << "\tReach(s, s'): " << reach.size() << std::endl;
    }

    // Check our variable count is correct
    assert(wr.nvars() == variables.selecteds.size() + variables.goods.size() + variables.vs.size() + reach.size());

    /////// CNF constraints ///////
    // [1] For each expanded state s, post a constraint OR_{s' solvable child of s} Good(s, s')
    //for (const auto s:sample_.alive_states()) {
    for (const auto s:sample_.expanded_states()) {
        cnfclause_t clause;
        for (unsigned sprime:sample_.successors(s)) {
            auto tx = get_transition_id(s, sprime);
            if (is_necessarily_bad(tx)) continue; // includes alive-to-dead transitions

            // Push it into the clause
            clause.push_back(Wr::lit(variables.goods.at(get_representative_id(tx)), true));
        }

        // Add clauses (1) for this state
        if (clause.empty()) {
            throw std::runtime_error(
                    "State #" + std::to_string(s) + " is marked as alive, but has no successor that can be good. "
                    "This is likely due to the feature pool not being large enough to distinguish some dead state from "
                    "some alive state. Try increasing the feature complexity bound");
        }
        wr.cl(clause);
        ++n_good_tx_clauses;
    }


    for (const auto s:sample_.expanded_states()) {
        for (const auto sprime:sample_.successors(s)) {
            if (is_necessarily_bad(get_transition_id(s, sprime))) continue; // includes alive-to-dead transitions
//                if (!sample_.is_alive(sprime)) continue;
            if (sample_.is_goal(sprime)) continue;
            if (!sample_.is_expanded(sprime)) continue;

            const auto good_s_prime = variables.goods.at(get_class_representative(s, sprime));

            for (unsigned dprime=1; dprime < max_d; ++dprime) {
                // (2) Good(s, s') and V(s',dprime) -> V(s) > dprime
                cnfclause_t clause{Wr::lit(good_s_prime, false),
                                   Wr::lit(variables.vs.at({sprime, dprime}), false)};

                for (unsigned d = dprime + 1; d <= max_d; ++d) {
                    clause.push_back(Wr::lit(variables.vs.at({s, d}), true));
                }
                wr.cl(clause);
                ++n_descending_clauses;
            }

            // (2') Border condition: V(s', D) implies -Good(s, s')
            wr.cl({Wr::lit(variables.vs.at({sprime, max_d}), false), Wr::lit(good_s_prime, false)});
            ++n_descending_clauses;
        }
    }

    // Clauses (6), (7):
    auto transitions_to_distinguish = distinguish_all_transitions();
    if (options.verbosity>0) {
        std::cout << "Posting distinguishability constraints for " << transitions_to_distinguish.size()
                  << " pairs of transitions" << std::endl;
    }
    for (const auto& tpair:transitions_to_distinguish) {
        assert (!is_necessarily_bad(tpair.tx1));
        const auto& [s, sprime] = get_state_pair(tpair.tx1);
        const auto& [t, tprime] = get_state_pair(tpair.tx2);

        cnfclause_t clause{Wr::lit(variables.goods.at(tpair.tx1), false)};

        // Compute first the Selected(f) terms
        for (feature_t f:compute_d1d2_distinguishing_features(feature_ids, sample_, s, sprime, t, tprime)) {
            clause.push_back(Wr::lit(variables.selecteds.at(f), true));
        }

        if (!is_necessarily_bad(tpair.tx2)) {
            auto good_t_tprime = variables.goods.at(tpair.tx2);
            clause.push_back(Wr::lit(good_t_tprime, true));
        }
        wr.cl(clause);
        n_separation_clauses += 1;
    }

    // (8): Force D1(s1, s2) to be true if exactly one of the two states is a goal state
    if (options.distinguish_goals) {
        for ( const auto s : sample_.full_training_set().all_alive()) {
            for( const auto t : sample_.full_training_set().all_goals()) {
        //for (unsigned s:sample_.goal_states()) {
            //for (unsigned t:sample_.nongoal_states()) {
                const auto d1feats = compute_d1_distinguishing_features(sample_, s, t);
                if (d1feats.empty()) {
                    undist_goal_warning(s, t);
                }

                cnfclause_t clause;
                for (unsigned f:d1feats) {
                    clause.push_back(Wr::lit(variables.selecteds.at(f), true));
                }

                wr.cl(clause);
                n_goal_clauses += 1;
            }
        }
    }
    
    if (!options.validate_features.empty()) {
        // If we only want to validate a set of features, we just force the Selected(f) to be true for them,
        // plus we don't really need any soft constraints.
//        std::cout << "Enforcing " << feature_ids.size() << " feature selections and ignoring soft constraints" << std::endl;
        for (unsigned f:feature_ids) {
            wr.cl({Wr::lit(variables.selecteds[f], true)});
        }
    } else {
//        std::cout << "Posting (weighted) soft constraints for " << variables.selecteds.size() << " features" << std::endl;
        for (unsigned f:feature_ids) {
            wr.cl({Wr::lit(variables.selecteds[f], false)}, sample_.feature_weight(f));
        }
    }

    n_selected_clauses += feature_ids.size();

    if (options.verbosity>0) {
        // Print a breakdown of the clauses
        std::cout << "A total of " << wr.nclauses() << " clauses were created" << std::endl;
        std::cout << "\tPolicy completeness [1]: " << n_good_tx_clauses << std::endl;
        std::cout << "\tTransition separation [5,6]: " << n_separation_clauses << std::endl;
        std::cout << "\tV descending along good transitions [X]: " << n_descending_clauses << std::endl;
        std::cout << "\tV is total function within bounds [X]: " << n_v_function_clauses << std::endl;
        std::cout << "\tGoal separation [X]: " << n_goal_clauses << std::endl;
        std::cout << "\t(Weighted) Select(f): " << n_selected_clauses << std::endl;
        assert(wr.nclauses() == n_selected_clauses + n_good_tx_clauses + n_descending_clauses
                                + n_v_function_clauses + n_separation_clauses
                                + n_goal_clauses);
    }
    return {CNFGenerationOutput::Success, variables};
}

std::vector<transition_pair> D2LEncoding::distinguish_all_transitions() const {
    std::vector<transition_pair> transitions_to_distinguish;
    transitions_to_distinguish.reserve(class_representatives_.size() * class_representatives_.size());

    for (const auto tx1:class_representatives_) {
        if (is_necessarily_bad(tx1)) continue;
        for (const auto tx2:class_representatives_) {
            if (tx1 != tx2) {
                transitions_to_distinguish.emplace_back(tx1, tx2);
            }
        }
    }
    return transitions_to_distinguish;
}


DNFPolicy D2LEncoding::generate_dnf(const std::vector<std::pair<unsigned, unsigned>>& goods, const std::vector<unsigned>& selecteds) const {
    DNFPolicy dnf(selecteds);
    for (const auto& [s, sprime]:goods) {
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


DNFPolicy D2LEncoding::generate_dnf(const std::vector<unsigned>& goods, const std::vector<unsigned>& selecteds) const {
    std::vector<std::pair<unsigned, unsigned>> pairs;
    pairs.reserve(goods.size());
    for (const auto& tx:goods) {
        pairs.push_back(get_state_pair(tx));
    }
    return generate_dnf(pairs, selecteds);
}


DNFPolicy D2LEncoding::generate_dnf_from_solution(const VariableMapping& variables, const SatSolution& solution) const {
    // Let's parse the relevant bits of the CNF solution:
    std::vector<unsigned> selecteds, goods;
    for (unsigned f=0; f < variables.selecteds.size(); ++f) {
        auto varid = variables.selecteds[f];
        if (varid>0 && solution.assignment.at(varid)) selecteds.push_back(f);
    }

    for (const auto& [txid, varid]:variables.goods) {
        if (varid>0 && solution.assignment.at(varid)) {
            goods.push_back(txid);
//            std::cout << "Good(" << get_state_pair(txid).first << ", " << get_state_pair(txid).second << ")" << std::endl;
        }
    }

//    for (const auto& [sd, varid]:variables.vs) {
//        if (varid>0 && solution.assignment.at(varid)) {
//            std::cout << "V(" << sd.first << ") = " << sd.second << std::endl;
//        }
//    }
    return generate_dnf(goods, selecteds);
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
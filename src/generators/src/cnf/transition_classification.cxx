
#include "transition_classification.h"
#include "types.h"

#include <iostream>
#include <vector>
#include <unordered_map>

#include <boost/functional/hash.hpp>
#include <common/helpers.h>


namespace sltp::cnf {


using feature_value_t = FeatureMatrix::feature_value_t;

sltp::cnf::transition_denotation compute_transition_denotation(feature_value_t s_f, feature_value_t sprime_f) {
    int type_s = (int) sprime_f - (int) s_f; // <0 if DEC, =0 if unaffected, >0 if INC
    int sign = (type_s > 0) ? 1 : ((type_s < 0) ? -1 : 0); // Get the sign
    return sltp::cnf::transition_denotation(bool(s_f > 0), sign);
}


void D2LEncoding::compute_equivalence_relations() {
    const auto& mat = tr_set_.matrix();

    // A mapping from a full transition trace to the ID of the corresponding equivalence class
    std::unordered_map<transition_trace, unsigned> from_trace_to_class_repr;

    for (const auto s:all_alive()) {
        for (unsigned sprime:successors(s)) {
            auto tx = std::make_pair((state_id_t) s, (state_id_t) sprime);
            auto id = (unsigned) transition_ids_inv_.size(); // Assign a sequential ID to the transition

            transition_ids_inv_.push_back(tx);
            auto it1 = transition_ids_.emplace(tx, id);
            assert(it1.second);

            // Store the type of the transition
            types_.push_back(is_solvable(sprime) ? transition_type::alive_to_solvable : transition_type::alive_to_dead);

            if (!is_solvable(sprime)) { // An alive-to-dead transition cannot be Good
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
                trace.denotations.emplace_back(compute_transition_denotation(mat.entry(s, f), mat.entry(sprime, f)));
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
    std::cout << "Number of transitions/equivalence classes: " << transition_ids_.size()
              << "/" << class_representatives_.size() << std::endl;
    std::cout << "Number of necessarily bad transitions/classes: " << necessarily_bad_transitions_.size()
              << "/" << necessarily_bad_classes.size() << std::endl;

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


std::vector<unsigned> D2LEncoding::
prune_dominated_features() {
    const auto& mat = tr_set_.matrix();

    std::vector<bool> dominated(nf_, false);

    if (!options.use_feature_dominance || !options.validate_features.empty())
        // No feature will be considered as dominated
        return feature_ids;

    std::cout << "Computing sets DT(f)... " << std::endl;
    std::vector<std::vector<transition_pair>> dt(nf_);
//    std::vector<boost::container::flat_set<transition_pair>> dt(nf_);
    for (unsigned f1 = 0; f1 < nf_; ++f1) {
        dt[f1] = compute_dt(f1);
    }

    std::cout << "Computing feature-dominance relations... " << std::endl;

    //! The computation and manipulation of the sets DT(f) below, notably the use of std::includes, requires that
    //! the vector of transition IDs is sorted
    assert(std::is_sorted(class_representatives_.begin(), class_representatives_.end()));

    unsigned ndominated = 0;
    for (unsigned f1 = 0; f1 < nf_; ++f1) {
        if (dominated[f1]) continue;

//        std::cout << "f=" << f1 << std::endl;
//        const auto d2_f1 = compute_dt(f1);
        const auto& d2_f1 = dt[f1];
        for (unsigned f2 = f1+1; f2 < nf_; ++f2) {
            if (dominated[f2]) continue;
            if (feature_weight(f1) > feature_weight(f2)) throw std::runtime_error("Features not ordered by complexity");

//            const auto d2_f2 = compute_dt(f2);
            const auto& d2_f2 = dt[f2];
            if (d2_f2.size() <= d2_f1.size() && std::includes(d2_f1.begin(), d2_f1.end(), d2_f2.begin(), d2_f2.end())) {
//                std::cout << "Feat. " << mat.feature_name(f1) << " dominates " << mat.feature_name(f2) << std::endl;
                ++ndominated;
                dominated[f2] = true;

            } else if (feature_weight(f1) == feature_weight(f2) && d2_f1.size() <= d2_f2.size()
                      && std::includes(d2_f2.begin(), d2_f2.end(), d2_f1.begin(), d2_f1.end())) {
//                std::cout << "Feat. " << mat.feature_name(f1) << " dominates " << mat.feature_name(f2) << std::endl;
                ++ndominated;
                dominated[f1] = true;
            }
        }
    }

    std::cout << "A total of " << ndominated << " features are dominated by some less complex feature and can be ignored" << std::endl;

    std::vector<unsigned> nondominated;

    for (unsigned f=0; f < nf_; f) {
        if (!dominated[f]) nondominated.push_back(f);
    }

    return nondominated;
}

std::vector<transition_pair> D2LEncoding::
compute_dt(unsigned f) {
    const auto& mat = tr_set_.matrix();

//    boost::container::flat_set<transition_pair> dt;
    std::vector<transition_pair> dt;

    for (const auto tx1:class_representatives_) {
        if (is_necessarily_bad(tx1)) continue;
        const auto& tx1pair = get_state_pair(tx1);
        const auto s = tx1pair.first;
        const auto sprime = tx1pair.second;


        for (const auto tx2:class_representatives_) {
            const auto& tx2pair = get_state_pair(tx2);
            const auto t = tx2pair.first;
            const auto tprime = tx2pair.second;

            if (are_transitions_d1d2_distinguished(
                    mat.entry(s, f), mat.entry(sprime, f), mat.entry(t, f), mat.entry(tprime, f))) {
                dt.emplace_back(tx1, tx2);
            }
        }
    }
//    std::cout << "DT(" << f << ") has " << dt.size() << " elements." << std::endl;
    return dt;
}



sltp::cnf::CNFGenerationOutput D2LEncoding::write(
        CNFWriter& wr, const std::vector<transition_pair>& transitions_to_distinguish)
{
    using Wr = CNFWriter;
    const auto& mat = tr_set_.matrix();
    const auto num_alive_transitions = transition_ids_.size();

    feature_ids = prune_dominated_features();

    auto varmapstream = get_ofstream(options.workspace + "/varmap.wsat");
    auto selected_map_stream = get_ofstream(options.workspace + "/selecteds.wsat");

    const unsigned max_d = compute_D();
    std::cout << "Using an upper bound for V_pi(s) values of " << max_d << std::endl;

    // Keep a map `good_tx_vars` from transition IDs to SAT variable IDs:
    std::unordered_map<unsigned, cnfvar_t> goods;

    // Keep a map from pairs (s, d) to SAT variable ID of the variable V(s, d)
    std::vector<std::vector<cnfvar_t>> vs(ns_);

    // Keep a map from each feature index to the SAT variable ID of Selected(f)
    std::vector<cnfvar_t> selecteds(nf_, std::numeric_limits<uint32_t>::max());

    unsigned n_select_vars = 0;
    unsigned n_v_vars = 0;
    unsigned n_descending_clauses = 0;
    unsigned n_v_function_clauses = 0;
    unsigned n_good_tx_clauses = 0;
    unsigned n_selected_clauses = 0;
    unsigned n_separation_clauses = 0;
    unsigned n_goal_clauses = 0;
    unsigned n_zero_clauses = 0;


    /////// CNF variables ///////
    // Create one "Select(f)" variable for each feature f in the pool
    for (unsigned f:feature_ids) {
        auto v = wr.var("Select(" + tr_set_.matrix().feature_name(f) + ")");
        selecteds[f] = v;
        selected_map_stream << f << "\t" << v << "\t" << tr_set_.matrix().feature_name(f) << std::endl;
        ++n_select_vars;
    }

    // Create variables V(s, d) variables for all all alive state s and d in 1..D
    for (const auto s:all_alive()) {
        const auto min_vs = get_vstar(s);
        const auto max_vs = get_max_v(s);
        assert(max_vs > 0 && max_vs <= max_d && min_vs <= max_vs);

        cnfclause_t within_range_clause;
        vs[s].reserve(max_d + 1);
        vs[s].push_back(std::numeric_limits<cnfvar_t>::max()); // We'll leave vs[s] 0 unused

        // TODO Note that we could be more efficient here and create only variables V(s,d) for those values of d that
        //  are within the bounds below. I'm leaving that as a future optimization, as it slightly complicates the
        //  formulation of constraints C2
        for (unsigned d = 1; d <= max_d; ++d) {
            const auto v_s_d = wr.var("V(" + std::to_string(s) + ", " + std::to_string(d) + ")");
            vs[s].push_back(v_s_d);
            n_v_vars += 1;

            if (d >= min_vs && d <= max_vs) {
                within_range_clause.push_back(Wr::lit(v_s_d, true));
            }
        }

        // Add clauses (4), (5)
        wr.cl(within_range_clause);
        n_v_function_clauses += 1;

        for (unsigned d = 1; d <= max_d; ++d) {
            for (unsigned dprime = d+1; dprime <= max_d; ++dprime) {
                wr.cl({Wr::lit(vs[s][d], false), Wr::lit(vs[s][dprime], false)});
                n_v_function_clauses += 1;
            }
        }
    }

    // Create a variable "Good(s, s')" for each transition (s, s') such that s' is solvable and (s, s') is not in BAD
    for (unsigned tx=0; tx < num_alive_transitions; ++tx) {
        if (is_necessarily_bad(tx)) continue; // This includes  alive-to-dead transitions

        const auto& txpair = get_state_pair(tx);
        const auto s = txpair.first;
        const auto sprime = txpair.second;

        cnfvar_t good_s_sprime = 0;
        auto repr = get_representative_id(tx);
        if (tx == repr) { // tx is an equivalence class representative: create the Good(s, s') variable
            good_s_sprime = wr.var("Good(" + std::to_string(s) + ", " + std::to_string(sprime) + ")");
            auto it = goods.emplace(tx, good_s_sprime);
            assert(it.second); // i.e. the SAT variable Good(s, s') is necessarily new
            varmapstream << good_s_sprime << " " << s << " " << sprime << std::endl;

        } else {  // tx is represented by repr, no need to create a redundant variable
            good_s_sprime = goods.at(repr);
        }
    }

    // [1] For each alive state s, post a constraint OR_{s' solvable child of s} Good(s, s')
    for (const auto s:all_alive()) {
        cnfclause_t clause;
        for (unsigned sprime:successors(s)) {
            auto tx = get_transition_id(s, sprime);
            if (is_necessarily_bad(tx)) continue; // includes alive-to-dead transitions

            // Push it into the clause
            clause.push_back(Wr::lit(goods.at(get_representative_id(tx)), true));
        }

        // Add clauses (1) for this state
        if (clause.empty()) {
            throw std::runtime_error("State #" + std::to_string(s) + " has no successor that can be good");
        }
        wr.cl(clause);
        ++n_good_tx_clauses;
    }

    // From this point on, no more variables will be created. Print total count.
    std::cout << "A total of " << wr.nvars() << " variables were created" << std::endl;
    std::cout << "\tSelect(f): " << n_select_vars << std::endl;
    std::cout << "\tGood(s, s'): " << goods.size() << std::endl;
    std::cout << "\tV(s, d): " << n_v_vars << std::endl;

    // Check our variable count is correct
    assert(wr.nvars() == n_select_vars + goods.size() + n_v_vars);

    /////// Rest of CNF constraints ///////
    std::cout << "Generating CNF encoding for " << all_alive().size() << " alive states, "
              <<  transition_ids_.size() << " alive-to-solvable and alive-to-dead transitions and "
              << class_representatives_.size() << " transition equivalence classes" << std::endl;

    for (const auto s:all_alive()) {
        for (const auto sprime:successors(s)) {
            if (is_necessarily_bad(get_transition_id(s, sprime))) continue; // includes alive-to-dead transitions
            const auto good_s_prime = goods.at(get_class_representative(s, sprime));

            if (options.decreasing_transitions_must_be_good) {
                // (3') Border condition: if s' is a goal, then (s, s') must be good
                if (is_goal(sprime)) {
                    wr.cl({Wr::lit(good_s_prime, true)});
                    ++n_descending_clauses;
                }
            }

            if (!is_alive(sprime)) continue;

            for (unsigned dprime=1; dprime < max_d; ++dprime) {
                // (2) Good(s, s') and V(s',dprime) -> V(s) > dprime
                cnfclause_t clause{Wr::lit(good_s_prime, false),
                                   Wr::lit(vs[sprime][dprime], false)};

                for (unsigned d = dprime + 1; d <= max_d; ++d) {
                    clause.push_back(Wr::lit(vs[s][d], true));

                    // (3) V(s') < V(s) -> Good(s, s')
                    if (options.decreasing_transitions_must_be_good) {
                        wr.cl({Wr::lit(vs[s][d], false),
                                      Wr::lit(vs[sprime][dprime], false),
                                      Wr::lit(good_s_prime, true)});
                        ++n_descending_clauses;
                    }
                }
                wr.cl(clause);
                ++n_descending_clauses;
            }

            // (2') Border condition: V(s', D) implies -Good(s, s')
            wr.cl({Wr::lit(vs[sprime][max_d], false), Wr::lit(good_s_prime, false)});
            ++n_descending_clauses;
        }
    }

    // Clauses (6), (7):
    std::cout << "Posting distinguishability constraints for " << transitions_to_distinguish.size()
              << " pairs of transitions" << std::endl;
    for (const auto& tpair:transitions_to_distinguish) {
        assert (!is_necessarily_bad(tpair.tx1));
        const auto& tx1pair = get_state_pair(tpair.tx1);
        const auto s = tx1pair.first;
        const auto sprime = tx1pair.second;
        const auto& tx2pair = get_state_pair(tpair.tx2);
        const auto t = tx2pair.first;
        const auto tprime = tx2pair.second;

        cnfclause_t clause{Wr::lit(goods.at(tpair.tx1), false)};

        // Compute first the Selected(f) terms
        for (feature_t f:compute_d1d2_distinguishing_features(feature_ids, tr_set_, s, sprime, t, tprime)) {
            clause.push_back(Wr::lit(selecteds.at(f), true));
        }

        if (!is_necessarily_bad(tpair.tx2)) {
            auto good_t_tprime = goods.at(tpair.tx2);
            clause.push_back(Wr::lit(good_t_tprime, true));
        }
        wr.cl(clause);
        n_separation_clauses += 1;
    }

    // (8): Force D1(s1, s2) to be true if exactly one of the two states is a goal state
    if (options.distinguish_goals) {
        for (unsigned s:goals_) {
            for (unsigned t:nongoals_) {

                if (!options.cross_instance_constraints &&
                    tr_set_.sample().state(s).instance_id() != tr_set_.sample().state(t).instance_id()) continue;

                const auto d1feats = compute_d1_distinguishing_features(tr_set_, s, t);
                if (d1feats.empty()) {
                    undist_goal_warning(s, t);
                    return sltp::cnf::CNFGenerationOutput::UnsatTheory;
                }

                cnfclause_t clause;
                for (unsigned f:d1feats) {
                    clause.push_back(Wr::lit(selecteds.at(f), true));
                }

                wr.cl(clause);
                n_goal_clauses += 1;
            }
        }
    }
    
    if (!options.validate_features.empty()) {
        // If we only want to validate a set of features, we just force the Selected(f) to be true for them,
        // plus we don't really need any soft constraints.
        std::cout << "Enforcing " << feature_ids.size() << " feature selections and ignoring soft constraints" << std::endl;
        for (unsigned f:feature_ids) {
            wr.cl({Wr::lit(selecteds[f], true)});
        }
    } else {
        std::cout << "Posting (weighted) soft constraints for " << selecteds.size() << " features" << std::endl;
        for (unsigned f:feature_ids) {
            wr.cl({Wr::lit(selecteds[f], false)}, feature_weight(f));
        }
    }

    n_selected_clauses += feature_ids.size();

    // Print a breakdown of the clauses
    std::cout << "A total of " << wr.nclauses() << " clauses were created" << std::endl;
    std::cout << "\t(Weighted) Select(f): " << n_selected_clauses << std::endl;
    std::cout << "\tPolicy completeness [1]: " << n_good_tx_clauses << std::endl;
    std::cout << "\tV descending along good transitions [2]: " << n_descending_clauses << std::endl;
    std::cout << "\tV is total function within bounds [3,4]: " << n_v_function_clauses << std::endl;
    std::cout << "\tTransition-separation clauses [5,6]: " << n_separation_clauses << std::endl;
    std::cout << "\tGoal clauses [7]: " << n_goal_clauses << std::endl;
//    std::cout << "\tZero clauses [8]: " << n_zero_clauses << std::endl;
    assert(wr.nclauses() == n_selected_clauses + n_good_tx_clauses + n_descending_clauses
                            + n_v_function_clauses + n_separation_clauses
                            + n_goal_clauses + n_zero_clauses);


    return sltp::cnf::CNFGenerationOutput::Success;
}

CNFGenerationOutput D2LEncoding::refine_theory(CNFWriter& wr) {
    if (!options.validate_features.empty()) {
        // If validate_features not empty, we will want to generate a validation CNF T(F', S) with all transitions
        // but with the given subset of features F' only.
        return write(wr, distinguish_all_transitions());
    }


    flaw_index_t flaws;
    bool previous_solution = check_existing_solution_for_flaws(flaws);
    if (previous_solution && flaws.empty()) {
        return CNFGenerationOutput::ValidationCorrectNoRefinementNecessary;
    }

    std::vector<transition_pair> transitions;

    if (options.use_incremental_refinement) {
        transitions = compute_transitions_to_distinguish(previous_solution, flaws);
        store_transitions_to_distinguish(transitions);
    } else {
        transitions = distinguish_all_transitions();
    }

    return write(wr, transitions);
}

std::vector<transition_pair> D2LEncoding::distinguish_all_transitions() const {
    std::vector<transition_pair> transitions_to_distinguish;
    transitions_to_distinguish.reserve(class_representatives_.size() * class_representatives_.size());
    const auto& sample = tr_set_.sample();

    for (const auto tx1:class_representatives_) {
        if (is_necessarily_bad(tx1)) continue;

        const auto& tx1pair = get_state_pair(tx1);
        const auto s = tx1pair.first;

        for (const auto tx2:class_representatives_) {
            const auto& tx2pair = get_state_pair(tx2);
            const auto t = tx2pair.first;

            if (sample.state(s).instance_id() != sample.state(t).instance_id()) continue;
            transitions_to_distinguish.emplace_back(tx1, tx2);
        }
    }
    return transitions_to_distinguish;
}

std::vector<transition_pair>
D2LEncoding::compute_transitions_to_distinguish(
        bool load_transitions_from_previous_iteration,
        const flaw_index_t& flaws) const {


    // Load the transitions from the last iteration, either from disk, or bootstrapping them
    std::vector<transition_pair> last_transitions;
    if (load_transitions_from_previous_iteration) {
        last_transitions = load_transitions_to_distinguish();
    } else {
        last_transitions = generate_t0_transitions();
    }

    std::vector<transition_pair> transitions_to_distinguish;

    std::vector<std::unordered_set<uint32_t>> index(transition_ids_.size());
    for (auto txpair:last_transitions) {
        index.at(txpair.tx1).insert(txpair.tx2);
    }
    for (const auto& it:flaws) {
        for (auto tx2:it.second) {
            index.at(it.first).insert(tx2);
        }
    }

    for (unsigned tx1=0; tx1<index.size(); ++tx1) {
        for (auto tx2:index[tx1]) {
            transitions_to_distinguish.emplace_back(tx1, tx2);
        }
    }
    return transitions_to_distinguish;
}

bool D2LEncoding::check_existing_solution_for_flaws(flaw_index_t& flaws)  const {
    auto ifs_good_transitions = get_ifstream(options.workspace + "/good_transitions.io");
    auto ifs_good_features = get_ifstream(options.workspace + "/good_features.io");

    std::vector<unsigned> good_features;
    int featureid = -1;
    while (ifs_good_features >> featureid) {
        good_features.push_back(featureid);
    }

    std::vector<unsigned> good_transitions_repr;
    int s = -1, sprime = -1;
    while (ifs_good_transitions >> s >> sprime) {
        good_transitions_repr.emplace_back(get_transition_id(s, sprime));
    }

    ifs_good_transitions.close();
    ifs_good_features.close();

    if (good_features.empty()) {
        return false;
    }

    // Let's exploit the equivalence classes between transitions. The transitions that have been read off as Good from
    // the SAT solution are already class representatives by definition of the SAT theory
    std::vector<unsigned> bad_transitions_repr;
    std::unordered_set<unsigned> good_set(good_transitions_repr.begin(), good_transitions_repr.end());

    for (auto repr:class_representatives_) {
        if (good_set.find(repr) == good_set.end()) {
            bad_transitions_repr.push_back(repr);
        }
    }


    // Let's consider pairs of transitions where the first one is Good, the second one is Bad, and (optionally)
    // both transitions belong to the same instance, and check whether they can be distinguished with the current
    // selection of features
    unsigned num_flaws = 0;
    const auto ntxs = transition_ids_.size();
    const auto& sample = tr_set_.sample();
    for (unsigned tx1=0; tx1 < ntxs; ++tx1) {
        const auto repr1 = get_representative_id(tx1);
        if (good_set.find(repr1) == good_set.end()) continue; // make sure tx2 is good

        const auto& tx1pair = get_state_pair(tx1);
        const auto inst1 = sample.state(tx1pair.first).instance_id();


        for (unsigned tx2=0; tx2 < ntxs; ++tx2) {
            const auto repr2 = get_representative_id(tx2);
            if (good_set.find(repr2) != good_set.end()) continue; // make sure tx2 is bad

            const auto& tx2pair = get_state_pair(tx2);
            const auto inst2 = sample.state(tx2pair.first).instance_id();

            if (!options.cross_instance_constraints && inst1 != inst2) continue;

            if (!are_transitions_d1d2_distinguishable(
                    tx1pair.first, tx1pair.second, tx2pair.first, tx2pair.second, good_features)) {
                // We found a pair of good/bad transitions that cannot be distinguished based on the selected features.
                flaws[repr1].push_back(repr2);
                ++num_flaws;
            }
        }
    }

    if (num_flaws) {
        std::cout << Utils::red() << "Refinement of computed policy found " << num_flaws << " flaws" << Utils::normal() << std::endl;
    } else {
        std::cout << Utils::green() << "Computed policy is correct with respect to full training set!" << Utils::normal() << std::endl;
    }

    return true;
}

bool D2LEncoding::are_transitions_d1d2_distinguishable(
        state_id_t s, state_id_t sprime, state_id_t t, state_id_t tprime, const std::vector<unsigned>& features) const {
    const auto& mat = tr_set_.matrix();
    for (unsigned f:features) {
        if (are_transitions_d1d2_distinguished(mat.entry(s, f), mat.entry(sprime, f),
                                               mat.entry(t, f), mat.entry(tprime, f))) {
            return true;
        }
    }
    return false;
}

void D2LEncoding::store_transitions_to_distinguish(
        const std::vector<transition_pair> &transitions) const {

    auto ofs = get_ofstream(options.workspace + "/last_iteration_transitions.io");
    for (const auto& txpair:transitions) {
        ofs << txpair.tx1 << " " << txpair.tx2 << std::endl;
    }
    ofs.close();
}


std::vector<transition_pair> D2LEncoding::load_transitions_to_distinguish() const {
    std::vector<transition_pair> transitions;
    auto ifs = get_ifstream(options.workspace + "/last_iteration_transitions.io");
    transition_id_t tx1, tx2;
    while (ifs >> tx1 >> tx2) {
        transitions.emplace_back(tx1, tx2);
    }
    ifs.close();
    return transitions;
}

std::vector<transition_pair> D2LEncoding::generate_t0_transitions(unsigned m) const {
    std::vector<transition_pair> transitions;

    std::mt19937 engine{std::random_device()()};
    std::uniform_int_distribution<int> dist(0, class_representatives_.size() - 1);

    for (const auto tx1:class_representatives_) {
        if (is_necessarily_bad(tx1)) continue;
        const auto& tx1pair = get_state_pair(tx1);
        const auto s = tx1pair.first;

        // Add all local transitions first
        for (unsigned sprime:successors(s)) {
            if (sprime == tx1pair.second) continue;
            transitions.emplace_back(tx1, get_class_representative(s, sprime));
        }

        // Add m randomly-chosen other transitions
        for (unsigned j = 0; j < m; ++j) {
            transitions.emplace_back(tx1, class_representatives_[dist(engine)]);
        }
    }

    return transitions;
}

} // namespaces

#include "sample.h"
#include <blai/sample.h>

#include <unordered_set>

namespace sltp::cnf {

std::vector<unsigned> randomize_int_sequence(std::mt19937& rng, unsigned n) {
    std::vector<unsigned> allints(n);
    std::iota(allints.begin(), allints.end(), 0);
    std::shuffle(allints.begin(), allints.end(), rng);
    return allints;
}

StateSpaceSample* sample_initial_states(std::mt19937& rng, const sltp::TrainingSet& trset, unsigned n) {
    const auto& transitions = trset.transitions();
    const auto& matrix = trset.matrix();

    auto allstates = randomize_int_sequence(rng, transitions.num_states());

    // Collect the first n alive states in the random order in allstates.
    std::vector<unsigned> sampled;
    for (const auto& s:allstates) {
        if (transitions.is_alive(s)) sampled.push_back(s);
        if (sampled.size() >= n) break;
    }

    return new StateSpaceSample(matrix, transitions, sampled);
}


bool evaluate_dnf(unsigned s, unsigned sprime, const DNFPolicy &dnf, const sltp::FeatureMatrix& matrix) {
    for (const auto& term:dnf.terms) {
        bool term_satisfied = true;

        for (const auto& [f, fval]:term) {
            const auto& fs = matrix.entry(s, f);
            const auto& fsprime = matrix.entry(sprime, f);

            if (fval != DNFPolicy::compute_state_value(fs) && fval != DNFPolicy::compute_transition_value(fs, fsprime)) {
                // one of the literals of the conjunctive term is not satisfied
                term_satisfied = false;
                break;
            }
        }
        if (term_satisfied) return true;
    }
    return false;
}


std::vector<unsigned>
find_flaws(std::mt19937& rng, const DNFPolicy& dnf, const StateSpaceSample& sample, unsigned batch_size) {
    std::vector<unsigned> flaws;
    // We go through all (alive) states in random order, and check whether the policy is defined for them.
    // If it is not, the policy is incomplete, so we mark the state as a policy flaw.
    const auto states = randomize_int_sequence(rng, sample.nstates_entire_sample());
    for (unsigned s:states) {
        if (!sample.is_alive(s)) continue;

        bool flawed = true;
        //std::cout << "Analyzing alive state " << s << "..." << std::endl;
        for (unsigned sprime:sample.successors(s)) {
            bool is_good = evaluate_dnf(s, sprime, dnf, sample.matrix());
            if (is_good) {
                if (!sample.is_unsolvable(sprime)) {
                    // A solvable-to-solvable transition predicted as good, the state is not flawed.
                    //std::cout << "Transition (" << s << ", " << sprime << ") was predicted as good" << std::endl;
                    flawed = false;
                }
                break;
            }
        }

        if (flawed) {
            //std::cout << "No transition defined as good for state " << s << ", adding it to flaw list" << std::endl;
            flaws.push_back(s);
        }

        if (flaws.size()>=batch_size) break;
    }

    //std::cout << "Flaw list: " << std::endl;
    //for (auto f:flaws) std::cout << f << ", ";
    //std::cout << std::endl;

    return flaws;
}


int select_action(unsigned s, const DNFPolicy& dnf, const StateSpaceSample& sample) {
    for (unsigned sprime:sample.successors(s)) {
        bool is_good = evaluate_dnf(s, sprime, dnf, sample.matrix());
        if (is_good) {
            return sprime;
        }
    }
    return -1;
}

std::vector<unsigned>
test_policy(std::mt19937& rng, const DNFPolicy& dnf, const StateSpaceSample& sample, unsigned batch_size) {
    std::vector<unsigned> flaws;
    std::vector<unsigned> roots;

    const auto states = randomize_int_sequence(rng, sample.nstates_entire_sample());
    for (unsigned s:states) {
        if (!sample.is_alive(s)) continue;
        roots.push_back(s);
    }


    for (const auto& root:roots) {
        unsigned current = root;

        std::unordered_set<unsigned> closed{current};
        std::vector<unsigned> path{current};

        while (!sample.is_goal(current)) {
            auto next = select_action(current, dnf, sample);
//            std::cout << "Policy advises transition (" << current << ", " << next << ")" << std::endl;

            if (next == -1) {
                // The policy is not defined on state "current"
//                std::cout << "Policy not defined on state " << current << std::endl;
                flaws.push_back(current);
                break;
            }

            if (sample.is_unsolvable(next)) {
                // The policy lead us into an unsolvable state
//                std::cout << "Policy reaches unsolvable state " << next << std::endl;
                flaws.push_back(current);
                break;
            }

            if (!closed.insert(next).second) {
                // We found a loop - mark all states in the loop as flaws
//                std::cout << "Policy reached loop on state " << next << std::endl;
                auto start = std::find(path.begin(), path.end(), next);
                flaws.insert(flaws.end(), start, path.end());

//                std::cout << "Loop: ";
//                for (auto it = std::find(path.begin(), path.end(), next); it != path.end(); ++it) {
//                    std::cout << *it << ", ";
//                }
//                std::cout << std::endl;

                break;
            }

            // The state is previously unseen
            path.push_back(next);
//            std::cout << "Path so far: ";
//            for (auto s:path) {
//                std::cout << s << ", ";
//            }
//            std::cout << std::endl;
            current = next;
        }

//        std::cout << std::endl;
//        std::cout << std::endl;
        if (flaws.size()>=batch_size) break;
    }

//    std::cout << "Flaw list: " << std::endl;
//    for (auto f:flaws) std::cout << f << ", ";
//    std::cout << std::endl;

    return flaws;
}


std::string print_term(const sltp::FeatureMatrix& matrix, const DNFPolicy::term_t& term, bool txt) {
    std::string res;
    unsigned i = 0;
    for (const auto& [f, fval]:term) {
        if (txt) {
            res += matrix.feature_name(f) + " " + print_feature_value(fval);
        } else {
            res += "[F" + std::to_string(f) + "]" + " " + print_feature_value(fval);
        }
        if (++i<term.size()) {
            res += ", ";
        }
    }
    return res;
}

void print_classifier(const sltp::FeatureMatrix& matrix, const DNFPolicy& dnf, const std::string& filename) {
    auto os = utils::get_ofstream(filename + ".dnf");
    auto ostxt = utils::get_ofstream(filename + ".txt");

    for (const auto& f:dnf.features) {
        os << f << " ";
    }
    os << std::endl;

    for (const auto& term:dnf.terms) {
        os << print_term(matrix, term, false) << std::endl;
        ostxt << print_term(matrix, term, true) << std::endl;
    }

    os.close();
    ostxt.close();
    std::cout << "DNF transition-classifier saved in " << filename << ".dnf and in " << filename << ".txt" << std::endl;
}



}


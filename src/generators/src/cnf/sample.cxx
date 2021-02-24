
#include "sample.h"
#include <blai/sample.h>

#include <unordered_set>

#include <boost/graph/strong_components.hpp>
#include <boost/graph/adjacency_list.hpp>

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


int select_action(unsigned s, const DNFPolicy& dnf, const StateSpaceSample& sample) {
    for (unsigned sprime:sample.successors(s)) {
        if (evaluate_dnf(s, sprime, dnf, sample.matrix())) {
            return (int) sprime;
        }
    }
    return -1;
}

std::vector<unsigned>
find_flaws(std::mt19937& rng, const DNFPolicy& dnf, const StateSpaceSample& sample, unsigned batch_size, bool verbose) {
    std::vector<unsigned> flaws;

    // We want to check the following over the  entire training set:
    //   (1) All alive states s have some outgoing transition (s, _) labeled as good
    //   (2) There is no alive-to-unsolvable transition labeled as good
    //   (3) There is no cycle among Good transitions

    // We randomize the order in which we check alive states
    auto alive = sample.full_training_set().all_alive(); // Make a non-const copy that we can shuffle
    std::shuffle(alive.begin(), alive.end(), rng);

    for (unsigned s:alive) {
        // Check (1)
        if (select_action(s, dnf, sample) == -1) {
            //std::cout << "No transition defined as good for state " << s << ", adding it to flaw list" << std::endl;
            flaws.push_back(s);
        }

        // Check (2)
        for (unsigned sprime:sample.successors(s)) {
            bool is_good = evaluate_dnf(s, sprime, dnf, sample.matrix());
            if (is_good && sample.is_unsolvable(sprime)) {
                flaws.push_back(s);
                break;
            }
        }

        if (flaws.size()>=batch_size) break;
    }

    // Check (3)
    const unsigned N = sample.full_training_set().num_states();

    using graph_t = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;
    graph_t graph(N);

    // Build graph
    for (unsigned s:alive) {
        for (unsigned sprime:sample.successors(s)) {
            if (sample.is_alive(sprime) && evaluate_dnf(s, sprime, dnf, sample.matrix())) {
                boost::add_edge(s, sprime, graph);
            }
        }
    }

    std::vector<int> component(N);
    boost::strong_components(graph, make_iterator_property_map(component.begin(), get(boost::vertex_index, graph), component[0]));

    std::vector<std::vector<unsigned>> cmp_to_states(N);
    for (unsigned s:alive) {
        cmp_to_states.at(component[s]).push_back(s);
    }

    for (const auto& component_states:cmp_to_states) {
        if (component_states.size()>1) {
            flaws.insert(flaws.end(), component_states.begin(), component_states.end());
        }
        if (flaws.size()>=batch_size) break;
    }

//    for (auto i = 0; i < component.size(); ++i) std::cout << "Vertex " << i << " in CC " << component[i] << std::endl;

    // Remove any excedent of flaws to conform to the required amount
    flaws.resize(std::min(flaws.size(), (std::size_t) batch_size));

    if (verbose) std::cout << "Flaw list: " << std::endl; for (auto f:flaws) std::cout << f << ", "; std::cout << std::endl;

    return flaws;
}


std::vector<unsigned>
test_policy(std::mt19937& rng, const DNFPolicy& dnf, const StateSpaceSample& sample, unsigned batch_size, bool verbose) {
    std::vector<unsigned> flaws;
    std::vector<unsigned> roots;

    const auto states = randomize_int_sequence(rng, sample.full_training_set().num_states());
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


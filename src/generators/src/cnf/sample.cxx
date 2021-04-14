
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


int select_action(unsigned s, const DNFPolicy& dnf, const TrainingSet& trset) {
    for (unsigned sprime:trset.transitions().successors(s)) {
        if (evaluate_dnf(s, sprime, dnf, trset.matrix())) {
            return (int) sprime;
        }
    }
    return -1;
}


void detect_cycles(const DNFPolicy& dnf, const TrainingSet& trset, unsigned batch_size, const std::vector<unsigned>& alive, std::vector<unsigned>& flaws) {
    const unsigned N = trset.transitions().num_states();

    using graph_t = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;
    graph_t graph(N);

    // Build graph
    for (unsigned s:alive) {
        for (unsigned sprime:trset.transitions().successors(s)) {
            if (trset.transitions().is_alive(sprime) && evaluate_dnf(s, sprime, dnf, trset.matrix())) {
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
}

std::vector<unsigned> StateSampler::randomize_all_alive_states(unsigned n) {
    auto alive = trset.transitions().all_alive(); // Make a non-const copy that we can shuffle
    std::shuffle(alive.begin(), alive.end(), rng);
    alive.resize(std::min(alive.size(), (std::size_t) n));
    return alive;
}

StateSpaceSample* RandomSampler::sample_initial_states(unsigned n) {
    return new StateSpaceSample(trset.matrix(), trset.transitions(), randomize_all_alive_states(n));
}


StateSpaceSample* FullSampleSampler::sample_initial_states(unsigned n) {
    std::vector<unsigned> all(trset.transitions().num_states());
    std::iota(all.begin(), all.end(), 0);
    return new StateSpaceSample(trset.matrix(), trset.transitions(), all);
}

std::vector<unsigned> RandomSampler::sample_flaws(const DNFPolicy& dnf, unsigned batch_size) {
    // We randomize the order in which we check alive states
    return sample_flaws(dnf, batch_size, randomize_all_alive_states());
}


StateSpaceSample* GoalDistanceSampler::sample_initial_states(unsigned n) {
    return new StateSpaceSample(trset.matrix(), trset.transitions(), randomize_and_sort_alive_states(n));
}


std::vector<unsigned> StateSampler::sample_flaws(const DNFPolicy& dnf, unsigned batch_size, const std::vector<unsigned>& states_to_check) {
    std::vector<unsigned> flaws;

    // We want to check the following over the  entire training set:
    //   (1) All alive states s have some outgoing transition (s, _) labeled as good
    //   (2) There is no alive-to-unsolvable transition labeled as good
    //   (3) There is no cycle among Good transitions
    for (unsigned s:states_to_check) {
        // Check (1)
        if (select_action(s, dnf, trset) == -1) {
            //std::cout << "No transition defined as good for state " << s << ", adding it to flaw list" << std::endl;
            flaws.push_back(s);
        }

        // Check (2)
        for (unsigned sprime:trset.transitions().successors(s)) {
            bool is_good = evaluate_dnf(s, sprime, dnf, trset.matrix());
            if (is_good && trset.transitions().is_unsolvable(sprime)) {
                flaws.push_back(s);
                break;
            }
        }

        if (flaws.size()>=batch_size) break;
    }

    if (verbosity>1) {
        std::cout << "Flaw list (incompleteness): " << std::endl; for (auto f:flaws) std::cout << f << ", "; std::cout << std::endl;
    }

    // Check (3)
    if (flaws.size()<batch_size) {
        detect_cycles(dnf, trset, batch_size, states_to_check, flaws);
    }

    // Remove any excedent of flaws to conform to the required amount
    flaws.resize(std::min(flaws.size(), (std::size_t) batch_size));


    if (verbosity>1) {
        std::cout << "Flaw list (loops): " << std::endl; for (auto f:flaws) std::cout << f << ", "; std::cout << std::endl;
    }

    return flaws;
}


std::vector<unsigned> GoalDistanceSampler::randomize_and_sort_alive_states(unsigned n) {
    auto sample = randomize_all_alive_states();
    // Sort states in ascending order of their min-distance to the goal V*(s)
    std::sort(sample.begin(), sample.end(), [&](const auto& lhs, const auto& rhs) {
        return trset.transitions_.vstar(lhs) < trset.transitions_.vstar(rhs);
    });
    sample.resize(std::min(sample.size(), (std::size_t) n));
    return sample;
}

std::unordered_map<unsigned, unsigned>
GoalDistanceSampler::compute_goal_distance_histogram(const std::vector<unsigned> states) {
    std::unordered_map<unsigned, unsigned> count;
    for (auto s:states) count[trset.transitions().vstar(s)]++;
    return count;
}


std::vector<unsigned> GoalDistanceSampler::sample_flaws(const DNFPolicy& dnf, unsigned batch_size) {
    // We look for samples in order of increasing distance to goal.
    return sample_flaws(dnf, batch_size, randomize_and_sort_alive_states());
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
    std::cout << "DNF transition-classifier saved in " << filename << ".txt" << std::endl;
}



}


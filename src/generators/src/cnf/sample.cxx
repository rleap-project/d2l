
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


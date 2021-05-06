
#include "sample.h"
#include <blai/sample.h>

#include <unordered_set>

namespace sltp::cnf {

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

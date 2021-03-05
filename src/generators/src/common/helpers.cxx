
#include <boost/algorithm/string.hpp>

#include <common/helpers.h>
#include <common/utils.h>


sltp::TransitionSample read_transition_data(const std::string& workspace, bool verbose) {
    std::string transitions_filename = workspace + "/transitions-info.io";
    std::cout << sltp::utils::blue() << "reading" << sltp::utils::normal() << " '" << transitions_filename << std::endl;
    auto ifs_transitions = sltp::utils::get_ifstream(transitions_filename);
    auto transitions = sltp::TransitionSample::read_dump(ifs_transitions, verbose);
    ifs_transitions.close();
    return transitions;
}


sltp::FeatureMatrix read_feature_matrix(const std::string& workspace, bool verbose) {
    std::string matrix_filename = workspace + "/feature-matrix.io";
    std::cout << sltp::utils::blue() << "reading" << sltp::utils::normal() << " '" << matrix_filename << std::endl;
    auto ifs_matrix = sltp::utils::get_ifstream(matrix_filename);
    auto matrix = sltp::FeatureMatrix::read_dump(ifs_matrix, verbose);
    ifs_matrix.close();
    return matrix;
}

sltp::Sample read_input_sample(const std::string& workspace) {
    auto ifs = sltp::utils::get_ifstream(workspace + "/sample.io");
    auto res = sltp::Sample::read(ifs);
    ifs.close();
    return res;
}

std::vector<std::string> read_nominals(const std::string& workspace) {
    auto ifs = sltp::utils::get_ifstream(workspace + "/nominals.io");
    std::string nominals_line;
    std::getline(ifs, nominals_line);
    ifs.close();

    if (nominals_line.empty()) return {};

    std::vector<std::string> nominals;
    boost::split(nominals, nominals_line, boost::is_any_of(" \t"));
    return nominals;
}


int transition_sign(int s_f, int sprime_f) {
    int type_s = sprime_f - s_f; // <0 if DEC, =0 if unaffected, >0 if INC
    return (type_s > 0) ? 1 : ((type_s < 0) ? -1 : 0);
}

bool are_transitions_d1d2_distinguished(int s_f, int sprime_f, int t_f, int tprime_f) {
    return (s_f == 0) != (t_f == 0) || transition_sign(s_f, sprime_f) != transition_sign(t_f, tprime_f);
}
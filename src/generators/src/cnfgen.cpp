#include <iostream>
#include <string>
#include <algorithm>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <blai/sample.h>
#include <blai/utils.h>
#include <cnf/generator.h>
#include <common/utils.h>
#include <cnf/transition_classification.h>
#include <common/helpers.h>


using namespace std;
namespace po = boost::program_options;

std::vector<unsigned> parse_id_list(const std::string& list) {
    std::vector<unsigned> ids;
    if (!list.empty()) {
        std::stringstream ss(list);
        while (ss.good()) {
            std::string substr;
            getline(ss, substr, ',');
            if (!substr.empty()) {
                ids.push_back((unsigned) atoi(substr.c_str()));
            }
        }
    }
    return ids;
}

//! Command-line option processing
sltp::cnf::Options parse_options(int argc, const char **argv) {
    sltp::cnf::Options options;

    po::options_description description("Generate a weighted max-sat instance from given feature "
                                        "matrix and transition samples.\n\nOptions");

    description.add_options()
        ("help,h", "Display this help message and exit.")
        ("verbose,v", "Show extra debugging messages.")

        ("workspace,w", po::value<std::string>()->required(),
                "Directory where the input files (feature matrix, transition sample) reside, "
                "and where the output .wsat file will be left.")

        ("enforce-features,e", po::value<std::string>()->default_value(""),
         "Comma-separated (no spaces!) list of IDs of feature we want to enforce in the abstraction.")

        ("validate-features", po::value<std::string>()->default_value(""),
         "Comma-separated (no spaces!) list of IDs of a subset of features we want to validate.")

        ("v_slack", po::value<double>()->default_value(2),
         "The slack value for the maximum allowed value for V_pi(s) = slack * V^*(s)")

        ("use-incremental-refinement",
         "In the D2L encoding, whether to use the incremental refinement approach")

        ("distinguish-goals",
         "In the D2L encoding, whether to post constraints to ensure distinguishability of goals")

        ("cross_instance_constraints",
         "In the D2L encoding, whether to post constraints to ensure distinguishability of goals "
         "and transitions coming from different training instances")

        ("encoding", po::value<std::string>()->default_value("d2l"),
             "The encoding to be used (options: {d2l}).")

        ("use-equivalence-classes",
         "In the D2L encoding, whether we want to exploit the equivalence relation "
         "among transitions given by the feature pool")

        ("decreasing_transitions_must_be_good",
         "In the D2L encoding, whether to force any V-descending transition to be labeled as Good")

        ("use-feature-dominance",
         "In the D2L encoding, whether we want to exploit the dominance among features to ignore "
         "dominated features and reduce the size of the encoding.")
    ;

    po::variables_map vm;

    try {
        po::store(po::command_line_parser(argc, argv).options(description).run(), vm);

        if (vm.count("help")) {
            std::cout << description << "\n";
            exit(0);
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cout << "Error with command-line options:" << ex.what() << std::endl;
        std::cout << std::endl << description << std::endl;
        exit(1);
    }

    options.workspace = vm["workspace"].as<std::string>();
    options.verbose = vm.count("verbose") > 0;
    options.use_equivalence_classes = vm.count("use-equivalence-classes") > 0;
    options.use_feature_dominance = vm.count("use-feature-dominance") > 0;
    options.use_incremental_refinement = vm.count("use-incremental-refinement") > 0;
    options.distinguish_goals = vm.count("distinguish-goals") > 0;
    options.decreasing_transitions_must_be_good = vm.count("decreasing_transitions_must_be_good") > 0;
    options.cross_instance_constraints = vm.count("cross_instance_constraints") > 0;
    options.v_slack = vm["v_slack"].as<double>();

    auto enc = vm["encoding"].as<std::string>();
    // ATM we only support one encoding in D2L, but let's leave the option open to other incoming encodings
    if  (enc == "d2l") options.encoding = sltp::cnf::Options::Encoding::D2L;
    else throw po::validation_error(po::validation_error::invalid_option_value, "encoding");


    // Split the comma-separated list of enforced feature IDS
    options.enforced_features = parse_id_list(vm["enforce-features"].as<std::string>());
//        for (auto x:enforced_features) std::cout << "\n" << x << std::endl;

    options.validate_features = parse_id_list(vm["validate-features"].as<std::string>());

    return options;
}

sltp::cnf::CNFGenerationOutput
write_encoding(CNFWriter& wr, const sltp::TrainingSet& sample, const sltp::cnf::Options& options) {
    assert(options.use_d2l_encoding());
    sltp::cnf::D2LEncoding generator(sample, options);
    return generator.refine_theory(wr);
}

int main(int argc, const char **argv) {
    float start_time = Utils::read_time_in_seconds();
    auto options = parse_options(argc, argv);

    // Read input training set
    sltp::TrainingSet trset(
            read_feature_matrix(options.workspace, options.verbose),
            read_transition_data(options.workspace, options.verbose),
            read_input_sample(options.workspace));

    std::cout << "Training sample: " << trset << std::endl;

    // We write the MaxSAT instance progressively as we generate the CNF. We do so into a temporary "*.tmp" file
    // which will be later processed by the Python pipeline to inject the value of the TOP weight, which we can
    // know only when we finish writing all clauses

    auto wsatstream = get_ofstream(options.workspace + "/theory.wsat.tmp");
    auto allvarsstream = get_ofstream(options.workspace + "/allvars.wsat");

    CNFWriter writer(wsatstream, &allvarsstream);
    auto output = write_encoding(writer, trset, options);

    wsatstream.close();
    allvarsstream.close();

    if (output != sltp::cnf::CNFGenerationOutput::ValidationCorrectNoRefinementNecessary) {
        // Write some characteristics of the CNF to a different file
        auto topstream = get_ofstream(options.workspace + "/top.dat");
        topstream << writer.top() << " " << writer.nvars() << " " << writer.nclauses();
        topstream.close();

        float total_time = Utils::read_time_in_seconds() - start_time;
        std::cout << "Total-time: " << total_time << std::endl;
        std::cout << "CNF Theory: " << writer.nvars() << " vars + " << writer.nclauses() << " clauses" << std::endl;
    }

    if (output == sltp::cnf::CNFGenerationOutput::UnsatTheory) {
        std::cout << Utils::warning() << "CNF theory is UNSAT" << std::endl;
    }

    return static_cast<std::underlying_type_t<sltp::cnf::CNFGenerationOutput>>(output);
}


#include "options.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>

namespace po = boost::program_options;

namespace sltp::cnf {


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
Options parse_options(int argc, const char **argv) {
    Options options;

    po::options_description description("Generate a weighted max-sat instance from given feature "
                                        "matrix and transition samples.\n\nOptions");

    description.add_options()
        ("help,h", "Display this help message and exit.")
        ("verbosity,v",  po::value<unsigned>()->default_value(0),
                "The level of verbosity - higher value means more verbosity.")

        ("workspace,w", po::value<std::string>()->required(),
         "Directory where the input files (feature matrix, transition sample) reside, "
         "and where the output .wsat file will be left.")

        ("validate-features", po::value<std::string>()->default_value(""),
         "Comma-separated (no spaces!) list of IDs of a subset of features we want to validate.")

        ("v_slack", po::value<unsigned>()->default_value(2),
         "The slack value for the maximum allowed value for V_pi(s) = slack * V^*(s)")

        ("distinguish-goals",
         "In the D2L encoding, whether to post constraints to ensure distinguishability of goals")

        ("use-equivalence-classes",
         "In the D2L encoding, whether we want to exploit the equivalence relation "
         "among transitions given by the feature pool")

        ("closed",
         "Whether to enforce policy-closedness constraints")

        ("optimal_steps", po::value<unsigned>()->default_value(0),
         "The upper bound to follow optimal transitions, i.e. Good(s,s') -> V^*(s') < V^*(s) if V^*(s) <= optimal_steps.")

        ("consistency_bound", po::value<unsigned>()->default_value(10),
          "For each V^*(s) <= consistency-bound, V(s') < V(s) and V^*(s) <= V(s) <= v_slack * V^*(s)")

        ("n_features", po::value<unsigned>()->default_value(0),
         "The number of features of the policy graph abstraction.")

        ("initial-sample-size", po::value<unsigned>()->default_value(100),
         "The number of solvable and dead states initially sampled at random.")

        ("solve", po::bool_switch()->default_value(false),
         "Whether we want to solve the maxsat problem or only generate the encoding.")

        ("refinement-batch-size", po::value<unsigned>()->default_value(10),
         "The number of flaws to add to the set sample on each iteration of the incremental approach.")

        ("seed", po::value<unsigned>()->default_value(0),
         "Random seed.")

        ("acyclicity", po::value<std::string>()->default_value("topological"),
         "The acyclicity encoding to be used (options: {topological, reachability, asp}).")

        ("encodings_dir", po::value<std::string>(), "The directory where the ASP encodings are.")
        ("sampling_strategy", po::value<std::string>()->default_value("random"),
          "The strategy to sample states when generating the encoding.")
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
    options.verbosity = vm["verbosity"].as<unsigned>();
    options.use_equivalence_classes = vm.count("use-equivalence-classes") > 0;
    options.distinguish_goals = vm.count("distinguish-goals") > 0;
    options.closed = vm.count("closed") > 0;
    options.optimal_steps = vm["optimal_steps"].as<unsigned>();
    options.consistency_bound = vm["consistency_bound"].as<unsigned>();
    options.n_features = vm["n_features"].as<unsigned>();
    options.v_slack = vm["v_slack"].as<unsigned>();
    options.solve = vm["solve"].as<bool>();
    options.initial_sample_size = vm["initial-sample-size"].as<unsigned>();
    options.refinement_batch_size = vm["refinement-batch-size"].as<unsigned>();
    options.seed = vm["seed"].as<unsigned>();
    options.validate_features = parse_id_list(vm["validate-features"].as<std::string>());
    options.encodings_dir = vm["encodings_dir"].as<std::string>();
    options.sampling_strategy = vm["sampling_strategy"].as<std::string>();
    options.acyclicity = vm["acyclicity"].as<std::string>();
    if (options.acyclicity != "reachability" &&
        options.acyclicity != "asp" &&
        options.acyclicity != "topological" &&
        options.acyclicity != "sd2l" ) {
        throw po::validation_error(po::validation_error::invalid_option_value, "acyclicity");
    }

    return options;
}


} // namespaces

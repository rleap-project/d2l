
#include <common/helpers.h>
#include <sltp/dl/cache.hxx>
#include <sltp/dl/factory.hxx>

#include <iostream>
#include <string>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


using namespace std;
namespace po = boost::program_options;

sltp::dl::Options parse_options(int argc, const char **argv) {
    po::options_description description("Generate a pool of non-redundante candidate features given"
                                        " a sample of states.\n\nOptions");

    description.add_options()
            ("help,h", "Display this help message and exit.")
            ("verbose,v", "Show extra debugging messages.")

            ("workspace,w", po::value<std::string>()->required(),
             "Directory where the input and output files will be expected / left.")

            ("timeout", po::value<int>()->default_value(-1), "The timeout, in seconds. Default: no timeout.")

            ("complexity-bound", po::value<unsigned>()->required(),
             "Maximum feature complexity for standard features.")

            ("dist-complexity-bound", po::value<unsigned>()->required(),
             "Maximum feature complexity for distance features.")

            ("cond-complexity-bound", po::value<unsigned>()->required(),
             "Maximum feature complexity for conditional features.")

            ("comparison-features", "Use comparison features of the type F1 < F2")

            ("generate-goal-concepts", "Whether to automatically generate goal-distinguishing concepts and roles.")
            ("print-denotations", "Whether print the denotations of all generated features.")
            ("print_hstar", "Print hstar value as a feature matrix column.")
            ("generate_or_concepts", "Generate DL disjunctions.")
            ("infty_feat_as_zero", "Interpret infinity values of features as zero.")
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

    sltp::dl::Options options;

    options.workspace = vm["workspace"].as<std::string>();
    options.timeout = vm["timeout"].as<int>();
    options.complexity_bound = vm["complexity-bound"].as<unsigned>();
    options.dist_complexity_bound = vm["dist-complexity-bound"].as<unsigned>();
    options.cond_complexity_bound = vm["cond-complexity-bound"].as<unsigned>();
    options.comparison_features = vm.count("comparison-features") > 0;
    options.generate_goal_concepts = vm.count("generate-goal-concepts") > 0;
    options.print_denotations = vm.count("print-denotations") > 0;
    options.print_hstar = vm.count("print_hstar") > 0;
    options.generate_or_concepts = vm.count("generate_or_concepts") > 0;
    options.infty_feat_as_zero = vm.count("infty_feat_as_zero") > 0;
    return options;
}

int main(int argc, const char **argv) {
    auto start_time = std::clock();
    const sltp::dl::Options options = parse_options(argc, argv);

    const sltp::Sample sample = read_input_sample(options.workspace);
    const std::vector<std::string> nominals = read_nominals(options.workspace);
    auto transitions = read_transition_data(options.workspace, false);

    sltp::dl::Factory factory(nominals, options);

    sltp::dl::Cache cache;
    factory.generate_basis(sample);

    std::vector<const sltp::dl::Concept*> goal_concepts;
    /*
    if (options.generate_goal_concepts) {
        goal_concepts = factory.generate_goal_concepts_and_roles(cache, sample);
    }
    */

    factory.generate_roles(cache, sample);

    auto concepts = factory.generate_concepts(cache, sample, start_time);
    factory.generate_features(concepts, cache, sample, transitions, goal_concepts);
//    factory.report_dl_data(cout);
    factory.log_all_concepts_and_features(concepts, cache, sample, options.workspace, options.print_denotations);

    // Print output feature matrix
    string output(options.workspace + "/feature-matrix.io");
    ofstream output_file(output);
    if( output_file.fail() ) throw runtime_error("Could not open filename '" + output + "'");
    factory.output_feature_matrix(output_file, cache, sample, transitions);

    return 0;
}


#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>

#include <blai/sample.h>
#include <cnf/generator.h>
#include <cnf/options.h>
#include <cnf/sample.h>
#include <cnf/transition_classification.h>
#include <common/utils.h>
#include <common/helpers.h>
#include <cnf/solver.h>


using namespace std;

namespace sltp::cnf {

std::tuple<CNFGenerationOutput, D2LEncoding, VariableMapping>
generate_maxsat_cnf(const StateSpaceSample& sample, const cnf::Options& options) {
    float start_time = sltp::utils::read_time_in_seconds();


    // We write the MaxSAT instance progressively as we generate the CNF. We do so into a temporary "*.tmp" file
    // which will be later processed by the Python pipeline to inject the value of the TOP weight, which we can
    // know only when we finish writing all clauses
    const auto tmpfilename = options.workspace + "/theory.wsat.tmp";
    auto otmpstream = sltp::utils::get_ofstream(tmpfilename);
    auto allvarsstream = sltp::utils::get_ofstream(options.workspace + "/allvars.wsat");

    CNFWriter writer(otmpstream, &allvarsstream);
    D2LEncoding generator(sample, options);
    auto [output, variables] = generator.write(writer);

    otmpstream.close();
    allvarsstream.close();

    if (output == cnf::CNFGenerationOutput::UnsatTheory) {
        std::cerr << utils::warning() << "CNF theory detected UNSAT while generating it" << std::endl;
        return {output, generator, variables};
    }

    // Now that we have finished printing all clauses of the encoding and have a final count of vars and clauses,
    // we add those relevant info to the final DIMACS file, and also iterate through the temp file replacing the value
    // of TOP where appropriate.
    std::cout << "Writing final DIMACS file..." << std::endl;
    auto dimacs = sltp::utils::get_ofstream(options.workspace + "/theory.wsat");
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    dimacs << "c WCNF model generated on " << std::put_time(&tm, "%Y%m%d %H:%M:%S") << std::endl;
    dimacs << "c Next line encodes: wcnf <nvars> <nclauses> <top>" << std::endl;
    dimacs << "p wcnf " << writer.nvars() << " " << writer.nclauses() << " " << writer.top() << std::endl;

    auto itmpstream = sltp::utils::get_ifstream(options.workspace + "/theory.wsat.tmp");
    std::string topstr = std::to_string(writer.top());
    std::string line;
    while (std::getline(itmpstream, line)) {
        boost::replace_all(line, "TOP" , topstr);
        dimacs << line << std::endl;
    }
    itmpstream.close();
    dimacs.close();

    // Once done, we can remove the temporary file
    remove(tmpfilename.c_str());
    float total_time = sltp::utils::read_time_in_seconds() - start_time;
    std::cout << "CNF Generation Time: " << total_time << std::endl;
    std::cout << "CNF Theory: " << writer.nvars() << " vars + " << writer.nclauses() << " clauses" << std::endl;
    return {output, generator, variables};
}

} // namespaces



int main(int argc, const char **argv) {
    float start_time = sltp::utils::read_time_in_seconds(),
            total_maxsat_time = 0,
            total_generation_time = 0;
    auto options = sltp::cnf::parse_options(argc, argv);

    std::mt19937 rng(options.seed);

    // Read input training set
    std::cout << "Parsing training data... " << std::endl;
    sltp::TrainingSet trset(
            read_feature_matrix(options.workspace, options.verbose),
            read_transition_data(options.workspace, options.verbose),
            read_input_sample(options.workspace));
    std::cout << "Done. Training sample: " << trset << std::endl;

    DNFPolicy dnf;

    if (options.verbose) {
        std::cout << "Sampling " << options.initial_sample_size << " alive states at random" << std::endl;
    }
    auto sample = std::unique_ptr<sltp::cnf::StateSpaceSample>(sltp::cnf::sample_initial_states(rng, trset, options.initial_sample_size));

    sltp::cnf::CNFGenerationOutput output;

    for (unsigned it=1; true; ++it) {
        std::cout << std::endl << std::endl << "###  STARTING ITERATION " << it << "  ###" << std::endl;

        if (options.verbose) {
            std::cout << *sample << std::endl;
        }

        float gent0 = sltp::utils::read_time_in_seconds();
        auto [o, encoder, variables] = generate_maxsat_cnf(*sample, options);
        total_generation_time += sltp::utils::read_time_in_seconds() - gent0;
        output = o;

        if (output != sltp::cnf::CNFGenerationOutput::Success) {
            std::cout << "Iteration #" << it << " failed to compute a correct DNF policy. "
                                                "Increase the feature complexity bound." << std::endl;
            break;
        }

        float solt0 = sltp::utils::read_time_in_seconds();
        auto solution = sltp::cnf::solve_cnf(options);
        total_maxsat_time += sltp::utils::read_time_in_seconds() - solt0;
        if (!solution.solved) {
            std::cout << "Theory is UNSAT." << std::endl;
            output = sltp::cnf::CNFGenerationOutput::UnsatTheory;
            break;
        }

        std::cout << "Maxsat solver found solution with cost " << solution.cost
                  << " in " << sltp::utils::read_time_in_seconds() - solt0 << "sec." << std::endl;
        dnf = encoder.generate_dnf_from_solution(variables, solution);
//            dnf = minimize_dnf();

        std::cout << "Features: " << std::endl;
        unsigned i = 0;
        for (unsigned f:dnf.features) {
            std::cout << "\t" << i++ << ": " << sample->matrix().feature_name(f)
                      << " [k=" << sample->matrix().feature_cost(f) << "]" << std::endl;
        }

//        auto flaws = find_flaws(rng, dnf, *sample, options.refinement_batch_size);
        auto flaws = test_policy(rng, dnf, *sample, options.refinement_batch_size);
        if (flaws.empty()) {
            std::cout << "Iteration #" << it << " found solution with no flaws" << std::endl;
            sltp::cnf::print_classifier(sample->matrix(), dnf, options.workspace + "/classifier");
            break;
        }
        sltp::cnf::print_classifier(sample->matrix(), dnf, options.workspace + "/classifier_" + std::to_string(it));
        std::cout << "Iteration #" << it << " found " << flaws.size() << " flaws" << std::endl;
        sample = std::unique_ptr<sltp::cnf::StateSpaceSample>(sample->add_states(flaws));

    }

    std::cout << "Total times across all iterations:" << std::endl;
    std::cout << "\tCNF Generation: " << total_generation_time << std::endl;
    std::cout << "\tMaxSAT Solver: " << total_maxsat_time << std::endl;
    std::cout << "\tTOTAL: " << sltp::utils::read_time_in_seconds() - start_time << std::endl;

    return static_cast<std::underlying_type_t<sltp::cnf::CNFGenerationOutput>>(output);
}


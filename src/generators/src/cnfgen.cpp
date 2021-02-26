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


void print_features(const StateSpaceSample& sample, const DNFPolicy& dnf) {
    cout << "Features: " << endl;
    unsigned i = 0;
    for (unsigned f:dnf.features) {
        cout << "\t" << i++ << ": " << sample.matrix().feature_name(f)
             << " [k=" << sample.matrix().feature_cost(f) << "]" << endl;
    }
}


struct TimeStats {
    float generation_time;
    float solution_time;
    TimeStats() : generation_time(0), solution_time(0) {}

};

class PolicyComputationStrategy {
public:
    virtual std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const StateSpaceSample& sample, TimeStats& time) = 0;
};

class SATPolicyComputationStrategy : public PolicyComputationStrategy {
public:
    std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const StateSpaceSample& sample, TimeStats& time) override {
        // Generate the encoding
        float gent0 = sltp::utils::read_time_in_seconds();
        const auto& [output, encoder, variables] = generate_maxsat_cnf(sample, options);
        time.generation_time += sltp::utils::read_time_in_seconds() - gent0;

        // If encoding already UNSAT, abort
        if (output != CNFGenerationOutput::Success) {
            std::cout << "Theory detected as UNSAT during generation." << std::endl;
            return {output, DNFPolicy()};
        }

        // Else try solving the encoding
        float solt0 = sltp::utils::read_time_in_seconds();
        auto solution = solve_cnf(options);
        auto tsolution = sltp::utils::read_time_in_seconds() - solt0;
        time.solution_time += tsolution;

        if (!solution.solved) {
            std::cout << "Theory detected as UNSAT by solver." << std::endl;
            return {CNFGenerationOutput::UnsatTheory, DNFPolicy()};
        }

        std::cout << "Solver found cost-" << solution.cost << " solution in " << tsolution << "sec." << std::endl;
        auto dnf = encoder.generate_dnf_from_solution(variables, solution);
//            dnf = minimize_dnf();
        return {CNFGenerationOutput::Success, dnf};
    }
};

class ASPPolicyComputationStrategy : public PolicyComputationStrategy {
public:
    std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const StateSpaceSample& sample, TimeStats& time) override {
        // Generate the encoding
        float gent0 = sltp::utils::read_time_in_seconds();
        D2LEncoding generator(sample, options);
        const auto instance = options.workspace + "/instance.lp";
        auto os = utils::get_ofstream(instance);
//        auto output = generator.generate_asp_instance_1(os);
        auto output = generator.generate_asp_instance_10(os);
        os.close();
        time.generation_time += sltp::utils::read_time_in_seconds() - gent0;

        // If encoding already UNSAT, abort
        if (output != CNFGenerationOutput::Success) {
            std::cout << "Theory detected as UNSAT during generation." << std::endl;
            return {output, DNFPolicy()};
        }

        // Else try solving the encoding
        float solt0 = sltp::utils::read_time_in_seconds();
        auto solution = solve_asp(
                options.encodings_dir + "/encoding10.lp",
                instance,
                options.workspace + "/clingo_output.log");
        auto tsolution = sltp::utils::read_time_in_seconds() - solt0;
        time.solution_time += tsolution;

        if (!solution.solved) {
            std::cout << "Theory detected as UNSAT by solver." << std::endl;
            return {CNFGenerationOutput::UnsatTheory, DNFPolicy()};
        }
        std::cout << "Solver found cost-" << solution.cost << " solution in " << tsolution << "sec." << std::endl;

        auto dnf = generator.generate_dnf(solution.goods, solution.selecteds);
//            dnf = minimize_dnf();
        return {CNFGenerationOutput::Success, dnf};
    }
};

std::unique_ptr<PolicyComputationStrategy> choose_strategy(const Options& options) {
    if (options.acyclicity == "asp") return std::make_unique<ASPPolicyComputationStrategy>();
    return std::make_unique<SATPolicyComputationStrategy>();
}

int run(const Options& options) {
    float start_time = sltp::utils::read_time_in_seconds();
    TimeStats time;

    // Initialize the random number generator with the given seed
    std::mt19937 rng(options.seed);

    // Read input training set
    std::cout << "Parsing training data... " << std::endl;
    sltp::TrainingSet trset(
            read_feature_matrix(options.workspace, options.verbose),
            read_transition_data(options.workspace, options.verbose),
            read_input_sample(options.workspace));
    std::cout << "Done. Training sample: " << trset << std::endl;

    if (options.verbose) {
        std::cout << "Sampling " << options.initial_sample_size << " alive states at random" << std::endl;
    }

    auto sampler = select_sampler(options.sampling_strategy, rng, trset, options.verbose);

    auto sample = std::unique_ptr<StateSpaceSample>(sampler->sample_initial_states(options.initial_sample_size));

    CNFGenerationOutput output;

    for (unsigned it=1; true; ++it) {
        std::cout << std::endl << std::endl << "###  STARTING ITERATION " << it << "  ###" << std::endl;
        if (options.verbose) std::cout << *sample << std::endl;

        auto strategy = choose_strategy(options);

        const auto& [output, dnf] = strategy->run(options, *sample, time);
        if (output != CNFGenerationOutput::Success) break;

        print_features(*sample, dnf);

        auto flaws = sampler->sample_flaws(dnf, options.refinement_batch_size);
//        auto flaws = test_policy(rng, dnf, *sample, options.refinement_batch_size);
        if (flaws.empty()) {
            std::cout << "Iteration #" << it << " found solution with no flaws" << std::endl;
            print_classifier(sample->matrix(), dnf, options.workspace + "/classifier");
            break;
        }

//        print_classifier(sample->matrix(), dnf, options.workspace + "/classifier_" + std::to_string(it));
        std::cout << "Iteration #" << it << " found " << flaws.size() << " flaws" << std::endl;
        sample = std::unique_ptr<StateSpaceSample>(sample->add_states(flaws));

    }

    std::cout << "Total times across all iterations:" << std::endl;
    std::cout << "\tCNF Generation: " << time.generation_time << std::endl;
    std::cout << "\tMaxSAT Solver: " << time.solution_time << std::endl;
    std::cout << "\tTOTAL: " << sltp::utils::read_time_in_seconds() - start_time << std::endl;

    return static_cast<std::underlying_type_t<CNFGenerationOutput>>(output);
}

} // namespaces


int main(int argc, const char **argv) {
    return run(sltp::cnf::parse_options(argc, argv));
}

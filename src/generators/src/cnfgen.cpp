#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>

#include <blai/sample.h>
#include <cnf/generator.h>
#include <cnf/options.h>
#include <cnf/sample.h>
#include <cnf/d2l.h>
#include <common/utils.h>
#include <common/helpers.h>
#include <cnf/solver.h>


using namespace std;

namespace sltp::cnf {



std::tuple<CNFGenerationOutput, VariableMapping>
generate_maxsat_cnf(D2LEncoding& generator, const TrainingSet& sample, const cnf::Options& options) {
    float start_time = utils::read_time_in_seconds();

    // We write the MaxSAT instance progressively as we generate the CNF. We do so into a temporary "*.tmp" file
    // which will be later processed by the Python pipeline to inject the value of the TOP weight, which we can
    // know only when we finish writing all clauses
    const auto tmpfilename = options.workspace + "/theory.wsat.tmp";
    auto otmpstream = utils::get_ofstream(tmpfilename);
    auto allvarsstream = utils::get_ofstream(options.workspace + "/allvars.wsat");

    CNFWriter writer(otmpstream, &allvarsstream);
    auto [output, variables] = generator.generate(writer);

    otmpstream.close();
    allvarsstream.close();

    if (output == cnf::CNFGenerationOutput::UnsatTheory) {
        std::cerr << utils::warning() << "CNF theory detected UNSAT while generating it" << std::endl;
        return {output, variables};
    }

    // Now that we have finished printing all clauses of the encoding and have a final count of vars and clauses,
    // we add those relevant info to the final DIMACS file, and also iterate through the temp file replacing the value
    // of TOP where appropriate.
    if (options.verbosity>1) {
        std::cout << "Writing final DIMACS file..." << std::endl;
    }
    auto dimacs = utils::get_ofstream(options.workspace + "/theory.wsat");
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    dimacs << "c WCNF model generated on " << std::put_time(&tm, "%Y%m%d %H:%M:%S") << std::endl;
    dimacs << "c Next line encodes: wcnf <nvars> <nclauses> <top>" << std::endl;
    dimacs << "p wcnf " << writer.nvars() << " " << writer.nclauses() << " " << writer.top() << std::endl;

    auto itmpstream = utils::get_ifstream(options.workspace + "/theory.wsat.tmp");
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
    float total_time = utils::read_time_in_seconds() - start_time;
    std::cout << "CNF [" << writer.nvars() << " vars, " << writer.nclauses()
              << " clauses] generated in " << total_time << " sec." << std::endl;

    return {output, variables};
}


void print_features(const FeatureMatrix& matrix, const DNFPolicy& dnf) {
    cout << "Features: " << endl;
    unsigned i = 0;
    for (unsigned f:dnf.features) {
        cout << "\t" << i++ << ": " << matrix.feature_name(f)
             << " [k=" << matrix.feature_cost(f) << "]" << endl;
    }
}


struct TimeStats {
    float generation_time;
    float solution_time;
    TimeStats() : generation_time(0), solution_time(0) {}

};

class PolicyComputationStrategy {
public:
    virtual std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const TrainingSet& sample, TimeStats& time) = 0;
};

class SATPolicyComputationStrategy : public PolicyComputationStrategy {
public:
    std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const TrainingSet& sample, TimeStats& time) override {
        // Generate the encoding
        float gent0 = utils::read_time_in_seconds();

        D2LEncoding encoder(sample, options);
        const auto& [output, variables] = generate_maxsat_cnf(encoder, sample, options);
        time.generation_time += utils::read_time_in_seconds() - gent0;

        // If encoding already UNSAT, abort
        if (output != CNFGenerationOutput::Success) {
            std::cout << "Theory detected as UNSAT during generation." << std::endl;
            return {output, DNFPolicy()};
        }

        // Else try solving the encoding
        float solt0 = utils::read_time_in_seconds();

        auto solution = solve_cnf(options.workspace + "/theory.wsat", options.workspace + "/maxsat_solver_run.log", options.verbosity>1);
        auto tsolution = utils::read_time_in_seconds() - solt0;
        time.solution_time += tsolution;

        if (!solution.solved) {
            std::cout << "Theory detected as UNSAT by solver." << std::endl;
            return {CNFGenerationOutput::UnsatTheory, DNFPolicy()};
        }

        std::cout << "Solution with cost " << solution.cost << " found in " << tsolution << "sec." << std::endl;
        auto dnf = encoder.generate_dnf_from_solution(variables, solution);
//            dnf = minimize_dnf();

        return {CNFGenerationOutput::Success, dnf};
    }
};

int run(const Options& options) {
    float start_time = utils::read_time_in_seconds();
    TimeStats time;

    // Initialize the random number generator with the given seed
    std::mt19937 rng(options.seed);

    // Read input training set
    std::cout << "Parsing training data... " << std::endl;
    sltp::TrainingSet dataset(
            read_feature_matrix(options.workspace, options.verbosity),
            read_transition_data(options.workspace, options.verbosity));
    std::cout << "Done. Training sample: " << dataset << std::endl;

    SATPolicyComputationStrategy strategy;
    const auto& [output, dnf] = strategy.run(options, dataset, time);

    if (output == CNFGenerationOutput::Success) {
        if (options.verbosity>0) print_features(dataset.matrix(), dnf);
        print_classifier(dataset.matrix(), dnf, options.workspace + "/classifier");
    }

    auto finish_time = utils::read_time_in_seconds();
    std::cout << "Total times: ";
    std::cout << "Theory generation: " << time.generation_time;
    std::cout << ", Solver: " << time.solution_time;
    std::cout << ", TOTAL: " << finish_time - start_time << std::endl;

    return static_cast<std::underlying_type_t<CNFGenerationOutput>>(output);
}

} // namespaces


int main(int argc, const char **argv) {
    return run(sltp::cnf::parse_options(argc, argv));
}

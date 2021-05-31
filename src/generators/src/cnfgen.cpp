#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>

#include <blai/sample.h>
#include <cnf/generator.h>
#include <cnf/options.h>
#include <cnf/sample.h>
#include <cnf/d2l.h>
#include <cnf/encoding_factory.h>
#include <common/utils.h>
#include <common/helpers.h>
#include <cnf/solver.h>


using namespace std;

namespace sltp::cnf {

std::tuple<CNFGenerationOutput, VariableMapping>
generate_maxsat_cnf(D2LEncoding& generator, const StateSpaceSample& sample, const cnf::Options& options) {
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
        float gent0 = utils::read_time_in_seconds();
        auto encoder = EncodingFactory::create(sample, options);
        const auto& [output, variables] = generate_maxsat_cnf(*encoder, sample, options);
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
        auto dnf = encoder->generate_dnf_from_solution(variables, solution);
//            dnf = minimize_dnf();
        return {CNFGenerationOutput::Success, dnf};
    }
};

class ASPPolicyComputationStrategy : public PolicyComputationStrategy {
public:
    std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const StateSpaceSample& sample, TimeStats& time) override {
        // Generate the encoding
        float gent0 = utils::read_time_in_seconds();
        D2LEncoding generator(sample, options);
        const auto instance = options.workspace + "/instance.lp";
        auto os = utils::get_ofstream(instance);
//        auto output = generator.generate_asp_instance_1(os);
//        auto output = generator.generate_asp_instance_10(os);
        auto output = generator.generate_asp_instance_20(os);
        os.close();
        time.generation_time += utils::read_time_in_seconds() - gent0;

        // If encoding already UNSAT, abort
        if (output != CNFGenerationOutput::Success) {
            std::cout << "Theory detected as UNSAT during generation." << std::endl;
            return {output, DNFPolicy()};
        }

        // Else try solving the encoding
        float solt0 = utils::read_time_in_seconds();
        auto solution = solve_asp(
                options.encodings_dir + "/encoding21.lp",
                instance,
                options.workspace + "/clingo_output.log",
                options.verbosity>1);
        auto tsolution = utils::read_time_in_seconds() - solt0;
        time.solution_time += tsolution;

        if (!solution.solved) {
            std::cout << "Theory detected as UNSAT by solver." << std::endl;
            return {CNFGenerationOutput::UnsatTheory, DNFPolicy()};
        }
        std::cout << "Solution with cost " << solution.cost << " found in " << tsolution << "sec." << std::endl;

        auto dnf = generator.generate_dnf_from_explicit_solution(solution);
//            dnf = minimize_dnf();
        return {CNFGenerationOutput::Success, dnf};
    }
};

std::unique_ptr<PolicyComputationStrategy> choose_strategy(const Options& options) {
    if (options.acyclicity == "asp") return std::make_unique<ASPPolicyComputationStrategy>();
    return std::make_unique<SATPolicyComputationStrategy>();
}

int run(const Options& options) {
    float start_time = utils::read_time_in_seconds();
    TimeStats time;

    // Initialize the random number generator with the given seed
    std::mt19937 rng(options.seed);

    // Read input training set
    std::cout << "Parsing training data... " << std::endl;
    sltp::TrainingSet trset(
            read_feature_matrix(options.workspace, options.verbosity),
            read_transition_data(options.workspace, options.verbosity),
            read_input_sample(options.workspace));
    std::cout << "Done. Training sample: " << trset << std::endl;

    if (options.verbosity) {
        std::cout << "Sampling " << options.initial_sample_size << " alive states at random" << std::endl;
    }

/////////////
//    GoalDistanceSampler gds(rng, trset, options.verbosity);
//    auto count = gds.compute_goal_distance_histogram(trset.transitions().all_alive());
////    for (const auto& [v, c]:count) std::cout << v << ": " << c << std::endl;
//    for (unsigned v=count.size(); v>0; --v) std::cout << v << ": " << count.at(v) << std::endl;
//
//    throw std::runtime_error("DONE");
/////////////

    auto sampler = select_sampler(options.sampling_strategy, rng, trset, options.verbosity);

    auto sample = std::unique_ptr<StateSpaceSample>(sampler->sample_initial_states(options.initial_sample_size));

    CNFGenerationOutput output;

    for (unsigned it=1; true; ++it) {
        if (options.verbosity>0) {
            std::cout << std::endl << std::endl << "###  STARTING ITERATION " << it << "  ###" << std::endl;
        } else {
            std::cout << std::endl << "Starting iteration " << it << std::endl;
        }

        if (options.verbosity) std::cout << *sample << std::endl;

        auto strategy = choose_strategy(options);

        const auto result = strategy->run(options, *sample, time);
        output = std::get<0>(result);
        const auto& dnf = std::get<1>(result);
        if (output != CNFGenerationOutput::Success) break;

        if (options.verbosity>0) {
            print_features(*sample, dnf);
        }

        auto flaws = sampler->sample_flaws(dnf, options.refinement_batch_size);
//        auto flaws = test_policy(rng, dnf, *sample, options.refinement_batch_size);
        if (flaws.empty()) {
            std::cout << "Solution found in iteration #" << it << " is correct!" << std::endl;
            print_classifier(sample->matrix(), dnf, options.workspace + "/classifier");
            break;
        }

//        print_classifier(sample->matrix(), dnf, options.workspace + "/classifier_" + std::to_string(it));
        std::cout << "Solution found in iteration #" << it << " has " << flaws.size() << " flaws" << std::endl;
        sample = std::unique_ptr<StateSpaceSample>(sample->add_states(flaws));

    }

    std::cout << "Total times: ";
    std::cout << "Theory generation: " << time.generation_time;
    std::cout << ", Solver: " << time.solution_time;
    std::cout << ", TOTAL: " << utils::read_time_in_seconds() - start_time << std::endl;

    return static_cast<std::underlying_type_t<CNFGenerationOutput>>(output);
}

} // namespaces


int main(int argc, const char **argv) {
    return run(sltp::cnf::parse_options(argc, argv));
}

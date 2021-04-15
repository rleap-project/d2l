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

class Exp2Latex{
    public:
        Exp2Latex( const std::string &file_name = "results.tex") : _iteration(0){
            _ofs = utils::get_ofstream(file_name);

            _header = std::string("\\begin{table*}[h!]\n") +
                      std::string("\\newcommand{\\ninst}{\\ensuremath{\\abs{P_i}}}\n") +
                      std::string("\\newcommand{\\tset}{\\ensuremath{\\S}}\n") +
                      std::string("\\newcommand{\\tseteq}{\\ensuremath{\\S/{\\scriptstyle\\sim}}}\n")+
                      std::string("\\newcommand{\\tall}{\\ensuremath{t_{\\text{all}}}}\n")+
                      std::string("\\newcommand{\\tsat}{\\ensuremath{t_{\\text{SAT}}}}\n")+
                      std::string("\\newcommand{\\npool}{\\ensuremath{\\abs{\\F}}}\n")+
                      std::string("\\newcommand{\\nfeats}{\\ensuremath{\\abs{\\Phi}}}\n")+
                      std::string("\\newcommand{\\sizepi}{\\ensuremath{\\abs{\\pi_\\Phi}}}\n")+
                      std::string("\\newcommand{\\cphi}{\\ensuremath{c_{\\Phi}}}\n")+
                      std::string("\\newcommand{\\maxk}{\\ensuremath{k^*}}\n")+
                      std::string("\\newcommand{\\dmax}{\\ensuremath{d_{max}}}\n")+
                      std::string("\\newcommand{\\dimens}{\\ensuremath{dim}}\n")+
                      std::string("\\newcommand{\\TK}{\\text{K}}\n")+
                      std::string("\\newcommand{\\TM}{\\text{M}}\n")+
                      std::string("\\centering\n")+
                      std::string("\\resizebox{\\textwidth}{!}{\n")+
                      std::string("\\begin{tabular}{LRRRRRRRRRRRRRRRRR}\n")+
                      std::string("\\toprule\n")+
                      std::string("D2L & \\ninst & \\text{Dim.} & \\text{Iter.}& \\tset & \\tseteq &\\dmax& \\npool & vars ") +
                      std::string("& clauses & \\tall &\\tsat& \\cphi & \\nfeats & \\maxk & \\sizepi \\\\ \n\\midrule\n");

            _bottom = std::string("\\bottomrule\n") +
                      std::string("\\end{tabular}\n}\n")+
                      std::string("\\caption{\\emph{Overview of D2L results}. ") +
                      std::string("\\ninst{} is number of training instances, and ")+
                      std::string("\\dimens{} is size of largest training instance along main generalization dimension(s). ") +
                      std::string("\\tset{} is number of sampled transitions in the training set, and ")+
                      std::string("\\tseteq{} is the number of distinguishable equivalence classes in \tset{}.")+
                      std::string("\\dmax{} is the max.\\ diameter of the training instances. ")+
                      std::string("\\npool{} is size of feature pool. ")+
                      std::string("``Vars'' and ``clauses'' are the number of variables and clauses in the (CNF form) of the theory $\\tsf$; ")+
                      std::string("\\tall{} is total CPU time, in sec., while ")+
                      std::string("\\tsat{} is CPU time spent solving  \\msat{} problems. ")+
                      std::string("\\cphi{} is optimal cost of SAT solution, ") +
                      std::string("\\nfeats{} is number of selected features, ") +
                      std::string("\\maxk{} is cost of the most complex feature in the policy, ")+
                      std::string("\\sizepi{}  is number of rules in the resulting policy. ")+
                      std::string("CPU times are given for the incremental constraint generation approach, ")+
                      std::string("which scales up better. }\n \\label{tab:experiments}\n \\end{table*}");
        }

        ~Exp2Latex(){
            if( _ofs.is_open() )
                _ofs.close();
        }

        void set_name( const std::string &name = "none" ){ _name = "\\Q_{" + name + "}"; }

        void set_n_instances( int i ){ _ninst = i; }

        void set_dim( const std::string &dim = "" ){ _dim = dim;}

        void add_tx( int t ){ _tset.emplace_back(t); }

        void add_tx_eq( int t ){ _tseteq.emplace_back(t); }

        void add_dmax( int dmax ){ _dmax.emplace_back(dmax); }

        void add_n_feat_pool( int n ){ _npool.emplace_back(n); }

        void add_sat_time( float t ){ _tsat.emplace_back(t); }

        void add_total_time( float t ){ _tall.emplace_back(t); }

        void add_total_cost( int c ){ _total_cost.emplace_back(c); }

        void add_max_cost_feat( int c ){ _max_cost_feat.emplace_back(c); }

        void add_n_feat( int n ){ _n_feat.emplace_back(n); }

        void add_size_pi( int s ){ _size_pi.emplace_back(s); }

        void add_vars( int v ){ _vars.emplace_back(v); }

        void add_clauses( int c ){ _clauses.emplace_back(c);}

        void next_iteration(){ _iteration++; }

        void pretty_printing( int d ){
            int base = 1;
            std::string units = "";
            if( d >= 1000000 ){ base = 100000; units = "\\TM";}
            else if( d >= 1000 ){ base = 100; units = "\\TK";}
            else{ _ofs << d; return; }
            _ofs << (d/(base*10)) << "." << ((d/base)%10) << units;
        }

        void print_to_file(){
            _ofs.setf(ios::fixed);
            _ofs.precision(1);
            _ofs << _header << std::endl;
            _ofs << _name << " & " << _ninst << " & " << _dim << " & ";
            for( unsigned i = 0; i < _iteration; i++ ) {
                if(i) _ofs << " & & & ";
                _ofs << i << " & " << _tset[i] << " & " << _tseteq[i] << " & ";
                _ofs << _dmax[i] << " & " << _npool[i] << " & ";
                pretty_printing(_vars[i]);
                _ofs << " & ";
                pretty_printing(_clauses[i]);
                _ofs << " & " << _tall[i] << " & " << _tsat[i] << " & ";
                _ofs << _total_cost[i] << " & " << _n_feat[i] << " & ";
                _ofs << _max_cost_feat[i] << " & " << _size_pi[i] << "\\\\" << std::endl;
            }
            _ofs << _bottom << std::endl;
        }
    private:
        std::ofstream _ofs;
        std::string _header;
        std::string _bottom;
        std::string _name;
        int _ninst;
        std::string _dim;
        std::vector< int > _tset;
        std::vector< int > _tseteq;
        std::vector< int > _dmax;
        std::vector< int > _npool;
        std::vector< int > _vars;
        std::vector< int > _clauses;
        std::vector< float > _tall;
        std::vector< float > _tsat;
        std::vector< int > _total_cost;
        std::vector< int > _n_feat;
        std::vector< int > _max_cost_feat;
        std::vector< int > _size_pi;
        int _iteration;
};



std::tuple<CNFGenerationOutput, VariableMapping>
generate_maxsat_cnf(D2LEncoding& generator, const StateSpaceSample& sample, const cnf::Options& options, Exp2Latex& t_tex ) {
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

    t_tex.add_tx( generator.get_num_tx() );
    t_tex.add_tx_eq( generator.get_num_tx_eq() );
    t_tex.add_n_feat_pool( generator.get_num_features() );
    t_tex.add_dmax( generator.compute_D() );
    t_tex.add_vars(writer.nvars());
    t_tex.add_clauses(writer.nclauses());

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
    virtual std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const StateSpaceSample& sample, TimeStats& time, Exp2Latex& t_tex ) = 0;
};

class SATPolicyComputationStrategy : public PolicyComputationStrategy {
public:
    std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const StateSpaceSample& sample, TimeStats& time, Exp2Latex& t_tex ) override {
        // Generate the encoding
        float gent0 = utils::read_time_in_seconds();
        auto encoder = EncodingFactory::create(sample, options);
        const auto& [output, variables] = generate_maxsat_cnf(*encoder, sample, options, t_tex );
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

        t_tex.add_sat_time( tsolution );

        if (!solution.solved) {
            std::cout << "Theory detected as UNSAT by solver." << std::endl;
            return {CNFGenerationOutput::UnsatTheory, DNFPolicy()};
        }

        std::cout << "Solution with cost " << solution.cost << " found in " << tsolution << "sec." << std::endl;
        auto dnf = encoder->generate_dnf_from_solution(variables, solution);
//            dnf = minimize_dnf();

        t_tex.add_total_cost( solution.cost );
        t_tex.add_size_pi( dnf.terms.size() );
        t_tex.add_n_feat( dnf.features.size() );
        unsigned max_cost = 0, c;
        for( const auto f : dnf.features ){
            c = encoder->get_feature_weight(f);
            if( c > max_cost ) max_cost = c;
        }
        t_tex.add_max_cost_feat( max_cost );

        return {CNFGenerationOutput::Success, dnf};
    }
};

class ASPPolicyComputationStrategy : public PolicyComputationStrategy {
public:
    std::pair<CNFGenerationOutput, DNFPolicy> run(const Options& options, const StateSpaceSample& sample, TimeStats& time, Exp2Latex& t_tex ) override {
        // Generate the encoding
        float gent0 = utils::read_time_in_seconds();
        D2LEncoding generator(sample, options);
        const auto instance = options.workspace + "/instance.lp";
        auto os = utils::get_ofstream(instance);
//        auto output = generator.generate_asp_instance_1(os);
        auto output = generator.generate_asp_instance_10(os);
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
                options.encodings_dir + "/encoding10.lp",
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
    auto t_tex = Exp2Latex(options.workspace + "/results.tex");
    // ToDo add the following info by options
    t_tex.set_name( options.exp_name ); // Given by options
    t_tex.set_n_instances( options.n_instances ); // Tx instances; Given by options
    t_tex.set_dim( options.dimensions ); // Given by options

    float start_time = utils::read_time_in_seconds();
    TimeStats time;

    // Initialize the random number generator with the given seed
    std::mt19937 rng(options.seed);

    // Read input training set
    std::cout << "Parsing training data... " << std::endl;
    sltp::TrainingSet trset(
            read_feature_matrix(options.workspace, options.verbosity),
            read_transition_data(options.workspace, options.verbosity));
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

        const auto result = strategy->run(options, *sample, time, t_tex);
        output = std::get<0>(result);
        const auto& dnf = std::get<1>(result);
        if (output != CNFGenerationOutput::Success) break;

        if (options.verbosity>0) {
            print_features(*sample, dnf);
        }

        auto flaws = sampler->sample_flaws(dnf, options.refinement_batch_size);
//        auto flaws = test_policy(rng, dnf, *sample, options.refinement_batch_size);

        t_tex.add_total_time( utils::read_time_in_seconds() - start_time );
        t_tex.next_iteration();

        if (flaws.empty()) {
            std::cout << "Solution found in iteration #" << it << " is correct!" << std::endl;
            print_classifier(sample->matrix(), dnf, options.workspace + "/classifier");
            break;
        }

//        print_classifier(sample->matrix(), dnf, options.workspace + "/classifier_" + std::to_string(it));
        std::cout << "Solution found in iteration #" << it << " has " << flaws.size() << " flaws" << std::endl;
        sample = std::unique_ptr<StateSpaceSample>(sample->add_states(flaws));

    }

    auto finish_time = utils::read_time_in_seconds();
    std::cout << "Total times: ";
    std::cout << "Theory generation: " << time.generation_time;
    std::cout << ", Solver: " << time.solution_time;
    std::cout << ", TOTAL: " << finish_time - start_time << std::endl;

    t_tex.print_to_file();

    return static_cast<std::underlying_type_t<CNFGenerationOutput>>(output);
}

} // namespaces


int main(int argc, const char **argv) {
    return run(sltp::cnf::parse_options(argc, argv));
}

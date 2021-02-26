
#include "solver.h"
#include "options.h"
#include <common/macros.h>

#include <common/utils.h>
#include <boost/lexical_cast.hpp>

#include <boost/algorithm/string.hpp>
#include <iostream>

namespace sltp::cnf {

int call(const std::string& cmd, bool verbose) {
    if (verbose) {
        std::cout << "Calling: " << cmd << std::endl;
    }
    auto res = system((cmd).c_str());
    if (verbose) {
        std::cout << "Call returned with exit code " << res << std::endl;
    }
    return res;
}

SatSolution solve_cnf(const std::string& cnf_filename, const std::string& output_filename, bool verbose) {
    call("open-wbo_static " + cnf_filename + " > " + output_filename, verbose);
    auto solutionf = utils::get_ifstream(output_filename);
    std::string line;

    SatSolution solution;

    while (std::getline(solutionf, line)) {
        const char code = line[0];
        std::string content = line.substr(std::min((unsigned) 2, (unsigned) line.size()));
        boost::trim(content);

        if (code == 'o') {
            solution.cost = std::min(solution.cost, boost::lexical_cast<int>(content));
        } else if (code == 's') {
            solution.result = content;
            if (content == "OPTIMUM FOUND" || content == "SATISFIABLE") {
                solution.solved = true;
            }
        } else if (code == 'v') {
            std::vector<std::string> literals;
            boost::split(literals, content, boost::is_any_of(" "));
            solution.assignment.resize(literals.size()+1);  // We'll have IDs going from 1 to literals.size+1
            for (const auto& lit:literals) {
                int val = boost::lexical_cast<int>(lit);
                if (val > 0) {
                    solution.assignment.at(val) = true;
                }
            }
        }
    }
    solutionf.close();
    return solution;
}


ASPSolution solve_asp(const std::string& domain_filename, const std::string& instance_filename, const std::string& output_filename, bool verbose) {
//    call("clingo " + domain_filename + " " + instance_filename + " > " + output_filename, options.verbosity>1);
    call("clingo -V0 --outf=1 " + domain_filename + " " + instance_filename + " | tee " + output_filename, verbose);
    auto solutionf = utils::get_ifstream(output_filename);
    std::string line;
    std::vector<std::string> atoms;

    ASPSolution solution;

    std::getline(solutionf, line);
    if (line == "ANSWER") {
        solution.solved = true;

        // the next line has the form "sel(9). good(1,0). sel(4). sel(12). good(2,5)."
        std::getline(solutionf, line);
        boost::split(atoms, line, boost::is_any_of(" "));
        std::vector<std::string> aux;
        for (const auto& atom:atoms) {
            boost::split(aux, atom, boost::is_any_of("()"));
            assert(aux.size()>=2);
            if (aux[0] == "sel") {
                solution.selecteds.push_back(boost::lexical_cast<unsigned>(aux[1]));
            } else if (aux[0] == "good") {
                boost::split(aux, aux[1], boost::is_any_of(","));
                solution.goods.emplace_back(boost::lexical_cast<unsigned>(aux[0]), boost::lexical_cast<unsigned>(aux[1]));
            }
        }

        // Next line: "COST 3"
        std::getline(solutionf, line);
        if (line.substr(0,4) == "COST") {
            solution.cost = boost::lexical_cast<unsigned>(line.substr(5, std::string::npos));
        }

        // Next line: "OPTIMUM"
        std::getline(solutionf, line);
        if (line != "OPTIMUM") {
            throw std::runtime_error("Clingo returned non-optimal solution.");
        }

    } else {
        solution.solved = false;
        if (line != "INCONSISTENT") {
            throw std::runtime_error("Unknown Clingo exit code:" + line);
        }
    }

    solutionf.close();
    return solution;
}

} // namespaces
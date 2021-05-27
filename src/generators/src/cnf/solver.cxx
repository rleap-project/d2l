
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
//    call("clasp --quiet=1 --configuration=jumpy  --parse-maxsat " + cnf_filename + " > " + output_filename, verbose);
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
            for (const auto& lit:literals) {
                int val = boost::lexical_cast<int>(lit);
                if (std::abs(val)+1>solution.assignment.size()) solution.assignment.resize(std::abs(val)+1);
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
//    call("clingo -V0 --outf=1 " + domain_filename + " " + instance_filename + " | tee " + output_filename, verbose);
    call("clingo -V0 --outf=1 " + domain_filename + " " + instance_filename + " >" + output_filename, verbose);
    auto solutionf = utils::get_ifstream(output_filename);
    std::string line;
    std::vector<std::string> atoms;

    ASPSolution solution;

    do { std::getline(solutionf, line); }
    while (line != "ANSWER" && line != "INONSISTENT" && !solutionf.eof());

    if (solutionf.eof()) throw std::runtime_error("Unexpected Clingo output. Check file: " + output_filename);

    if (line == "ANSWER") {
        solution.solved = true;

        // the next line has the form "sel(9, 1). eff(3,2,inc). pre(0,0,dontcare)..."
        std::getline(solutionf, line);
        boost::split(atoms, line, boost::is_any_of(" "));
        std::vector<std::string> aux;
        for (const auto& atom:atoms) {
            boost::split(aux, atom, boost::is_any_of("(),"));
            assert(aux.size()>=2);
            if (aux.at(0) == "sel") {
                auto feature = boost::lexical_cast<unsigned>(aux.at(1));
                auto position = boost::lexical_cast<unsigned>(aux.at(2));
                if (position>=solution.selecteds.size()) solution.selecteds.resize(position+1);
                solution.selecteds.at(position) = feature;
            } else if (aux.at(0) == "pre") {
                auto rule = boost::lexical_cast<unsigned>(aux.at(1));
                auto position = boost::lexical_cast<unsigned>(aux.at(2));
                auto value = aux.at(3);
                solution.pres_[rule].emplace_back(position, value);
            } else if (aux.at(0) == "eff") {
                auto rule = boost::lexical_cast<unsigned>(aux.at(1));
                auto position = boost::lexical_cast<unsigned>(aux.at(2));
                auto value = aux.at(3);
                solution.effs_[rule].emplace_back(position, value);
            }
        }

        // Next line: "COST 3"
        std::getline(solutionf, line);
        if (line.substr(0,4) == "COST") {
            solution.cost = boost::lexical_cast<int>(line.substr(5, std::string::npos));
        }

        // Next line: "OPTIMUM"
        std::getline(solutionf, line);
        if (line != "OPTIMUM") {
            throw std::runtime_error("Clingo returned non-optimal solution.");
        }

    }

    solutionf.close();
    return solution;
}

} // namespaces
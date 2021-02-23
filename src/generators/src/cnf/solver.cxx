
#include "solver.h"
#include "options.h"
#include <common/macros.h>

#include <common/utils.h>
#include <boost/lexical_cast.hpp>

#include <boost/algorithm/string.hpp>
#include <iostream>

namespace sltp::cnf {

SatSolution solve_cnf(const sltp::cnf::Options& options) {
    std::cout << "Calling OpenWBO on input file: " << options.workspace + "/theory.wsat" << std::endl;
//            ['open-wbo_static', PosixPath('/home/gfrances/projects/code/learning-deadend-classifiers/workspace/spanner.maxsat.k_14/theory.wsat')]
    auto res = system(("open-wbo_static " + options.workspace + "/theory.wsat > " + options.workspace +
                       "/maxsat_solver_run.log").c_str());
    UNUSED(res);
//    if (res != 0) {
//        throw std::runtime_error("Maxsat solver exited with return value of " + std::to_string(res));
//    }
    std::cout << "OpenWBO returned with exit code " << res << std::endl;

    auto solutionf = utils::get_ifstream(options.workspace + "/maxsat_solver_run.log");
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

} // namespaces

#pragma once

#include <limits>
#include <vector>
#include <string>

namespace sltp::cnf {
struct Options;

struct SatSolution {
    bool solved;
    int cost;
    std::vector<bool> assignment;
    std::string result;

    SatSolution() :
            solved(false), cost(std::numeric_limits<int>::max()), assignment(), result()
    {}
};

SatSolution solve_cnf(const sltp::cnf::Options& options);

}

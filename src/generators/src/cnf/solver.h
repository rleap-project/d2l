
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

    SatSolution() : solved(false), cost(std::numeric_limits<int>::max()), assignment(), result() {}
};

struct ASPSolution {
    bool solved;
    int cost;
    std::vector<std::pair<unsigned, unsigned>> goods;
    std::vector<unsigned> selecteds;

    ASPSolution() : solved(false), cost(std::numeric_limits<int>::max()), goods(), selecteds() {}
};


SatSolution solve_cnf(const std::string& cnf_filename, const std::string& output_filename, bool verbose);

ASPSolution solve_asp(const std::string& domain_filename, const std::string& instance_filename, const std::string& output_filename, bool verbose);

}

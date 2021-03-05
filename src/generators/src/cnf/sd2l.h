
#pragma once

#include <cnf/d2l.h>


namespace sltp::cnf {


class SD2LEncoding : public D2LEncoding {
public:
    SD2LEncoding(const StateSpaceSample& sample, const Options& options) : D2LEncoding(sample, options) {}

    std::pair<cnf::CNFGenerationOutput, VariableMapping> generate(CNFWriter& wr) override;

protected:

    inline void two_comparator(CNFWriter &wr, int &prefix_id, const cnfvar_t &x1, const cnfvar_t &x2, std::vector< cnfvar_t > &z );
    inline void merge_network(CNFWriter &wr, int &prefix_id, const std::vector< cnfvar_t > &y1,
                              const std::vector< cnfvar_t > &y2, std::vector< cnfvar_t > &y );
    inline void sorting_network(CNFWriter &wr, int &prefix_id, const std::vector< cnfvar_t > &x, std::vector< cnfvar_t > &y );
};

} // namespaces


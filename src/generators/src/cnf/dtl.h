
#pragma once

#include <cnf/d2l.h>


namespace sltp::cnf {
    class DTLEncoding : public D2LEncoding {
    public:
        DTLEncoding(const StateSpaceSample& sample, const Options& options) : D2LEncoding(sample, options) {}

        std::pair<cnf::CNFGenerationOutput, VariableMapping> generate(CNFWriter& wr) override;
    };

} // namespaces


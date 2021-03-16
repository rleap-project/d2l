#pragma once

#include "sd2l.h"
#include "dtl.h"
#include "d2l.h"

namespace sltp::cnf {
    class EncodingFactory {
    public:
        EncodingFactory();

        //! Factory method
        static std::unique_ptr<D2LEncoding> create(const StateSpaceSample &sample, const Options &options) {
            if (options.acyclicity == "sd2l") return std::make_unique<SD2LEncoding>(sample, options);
            else if (options.acyclicity == "dtl") return std::make_unique<DTLEncoding>(sample, options);
            return std::make_unique<D2LEncoding>(sample, options);
        }
    };
}
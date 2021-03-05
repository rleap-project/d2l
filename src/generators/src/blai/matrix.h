
#pragma once

#include <cassert>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>


namespace sltp {

class FeatureMatrix {
public:
    using feature_value_t = uint16_t;

protected:
    std::size_t num_features_;

    //! Contains pairs of feature name, feature cost
    std::vector<std::pair<std::string, unsigned>> feature_data_;
    std::unordered_map<std::string, unsigned> feature_name_to_id_;
    std::vector<std::vector<feature_value_t>> rowdata_;
    std::vector<bool> numeric_features_;


public:
    FeatureMatrix()
        : num_features_(0)
    {}

    FeatureMatrix(const FeatureMatrix&) = default;
    FeatureMatrix(FeatureMatrix&&) = default;

    virtual ~FeatureMatrix() = default;

    std::size_t num_states() const { return rowdata_.size(); }

    std::size_t num_features() const { return num_features_; }

    const std::string& feature_name(unsigned i) const {
        assert(i < num_features_);
        return feature_data_.at(i).first;
    }

    unsigned feature_cost(unsigned i) const {
        return feature_data_.at(i).second;
    }

    feature_value_t entry(unsigned s, unsigned f) const {
        return rowdata_[s][f];
    }

    feature_value_t operator()(unsigned s, unsigned f) const {
        return entry(s, f);
    }

    void print(std::ostream &os) const;

    // readers
    void read(std::ifstream &is);

    static FeatureMatrix read_dump(std::ifstream &is, bool verbose) {
        FeatureMatrix matrix;
        matrix.read(is);
        if (verbose) {
            std::cout << "FeatureMatrix::read_dump: "
                      << "#states=" << matrix.num_states() << ", #features=" << matrix.num_features() << std::endl;
        }
        return matrix;
    }
};

} // namespaces

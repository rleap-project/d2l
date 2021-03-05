
#pragma once

#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <common/utils.h>

namespace sltp {

class FeatureMatrix {
    public:
        using feature_value_t = uint16_t;

    protected:
        const std::size_t num_states_;
        const std::size_t num_features_;

        //! Contains pairs of feature name, feature cost
        std::vector<std::pair<std::string, unsigned>> feature_data_;
        std::unordered_map<std::string, unsigned> feature_name_to_id_;
        std::vector<std::vector<feature_value_t>> rowdata_;
        std::vector<bool> numeric_features_;


    public:
        FeatureMatrix(std::size_t num_states, std::size_t num_features)
                : num_states_(num_states),
                  num_features_(num_features),
                  numeric_features_(num_features_, false)
        {
        }

        FeatureMatrix(const FeatureMatrix&) = default;
        FeatureMatrix(FeatureMatrix&&) = default;

        virtual ~FeatureMatrix() = default;

        std::size_t num_states() const { return num_states_; }

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

        void print(std::ostream &os) const {
            auto nnumeric = std::count(numeric_features_.begin(), numeric_features_.end(), true);
            unsigned nbinary = num_features_ - nnumeric;

            os << "FeatureMatrix stats: #states=" << num_states_
               << ", #features=" << num_features_
               << ", #binary-features=" << nbinary
               << ", #numeric-features=" << nnumeric
               << std::endl;
            for (unsigned s = 0; s < num_states_; ++s) {
                os << "state " << s << ":";
                for (unsigned f = 0; f < num_features_; ++f) {
                    feature_value_t value = entry(s, f);
                    if (value > 0)
                        os << " " << f << ":" << value;
                }
                os << std::endl;
            }
        }

        // readers
        void read(std::ifstream &is) {
            std::string line;

            // read features
            for (unsigned i = 0; i < num_features_; ++i) {
                std::string feature;
                is >> feature;
                feature_name_to_id_.emplace(feature, feature_data_.size());
                feature_data_.emplace_back(feature, 0);
            }

            // read feature costs
            for (unsigned i = 0; i < num_features_; ++i) {
                unsigned cost = 0;
                is >> cost;
                assert(cost > 0);
                assert(feature_data_[i].second == 0);
                feature_data_[i].second = cost;
            }

            // Read the actual feature matrix data
            rowdata_.reserve(num_states_);
            feature_value_t value = 0;
            for (int i = 0; i < num_states_; ++i) {
                unsigned s = 0, nentries = 0;
                is >> s >> nentries;
                assert(i == s);  // Make sure states are listed in increasing order

                std::vector<feature_value_t> data(num_features_, 0);
                for(unsigned j = 0; j < nentries; ++j) {
                    char filler;
                    unsigned f = 0;
                    is >> f >> filler >> value;
                    assert(filler == ':');
                    assert(f < num_features_);
                    assert(value > 0);
                    data[f] = value;
                }
                rowdata_.push_back(std::move(data));
            }

            // Figure out which features are binary, which are numeric
            assert(numeric_features_.size() == num_features_);
            for (unsigned f = 0; f < num_features_; ++f) {
                bool has_value_other_than_0_1 = false;
                for (unsigned s = 0; s < num_states_; ++s) {
                    if (entry(s, f) > 1) {
                        has_value_other_than_0_1 = true;
                        break;
                    }
                }
                if (has_value_other_than_0_1) {
                    numeric_features_[f] = true;
                }
            }
        }

        static FeatureMatrix read_dump(std::ifstream &is, bool verbose) {
            unsigned num_states = 0, num_features = 0;
            is >> num_states >> num_features;
            FeatureMatrix matrix(num_states, num_features);
            matrix.read(is);
            if (verbose) {
                std::cout << "FeatureMatrix::read_dump: "
                          << "#states=" << matrix.num_states()
                          << ", #features=" << matrix.num_features()
                          << std::endl;
            }
            return matrix;
        }
    };

} // namespaces

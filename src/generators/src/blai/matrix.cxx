
#include <blai/matrix.h>

#include <limits>
#include <algorithm>
#include <fstream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


namespace sltp {

void FeatureMatrix::print(std::ostream &os) const {
    auto nnumeric = std::count(numeric_features_.begin(), numeric_features_.end(), true);
    unsigned nbinary = num_features_ - nnumeric;

    os << "FeatureMatrix stats: #states=" << rowdata_.size()
       << ", #features=" << num_features_
       << ", #binary-features=" << nbinary
       << ", #numeric-features=" << nnumeric
       << std::endl;
    for (unsigned s = 0; s < rowdata_.size(); ++s) {
        os << "state " << s << ":";
        for (unsigned f = 0; f < num_features_; ++f) {
            feature_value_t value = entry(s, f);
            if (value > 0)
                os << " " << f << ":" << value;
        }
        os << std::endl;
    }
}

void FeatureMatrix::read(std::ifstream &is) {
    std::string line;

    // Line 0: Header line, just ignore
    std::getline(is, line);

    // Line 1: feature names
    std::vector<std::string> names, complexities, values;
    std::getline(is, line);
    boost::split(names, line, boost::is_any_of(" "));
    num_features_ = names.size();

    // Line 2: feature complexities
    std::getline(is, line);
    boost::split(complexities, line, boost::is_any_of(" "));
    if (complexities.size() != num_features_) throw std::runtime_error("Input error in feature matrix");
    for (unsigned i = 0; i < num_features_; ++i) {
        feature_name_to_id_.emplace(names[i], feature_name_to_id_.size());
        feature_data_.emplace_back(names[i], boost::lexical_cast<unsigned>(complexities[i]));
    }

    // Rest of lines: one line per state, corresponding to the feature valuation of that state
    while (std::getline(is, line)) {
        boost::split(values, line, boost::is_any_of(" "));
        if (values.size() != num_features_) throw std::runtime_error("Input error in feature matrix");

        rowdata_.emplace_back(num_features_);
        auto& row = rowdata_.back();
        for (unsigned i= 0; i < num_features_; ++i) {
            auto parsed = boost::lexical_cast<int>(values[i]);
            row[i] = (parsed == std::numeric_limits<int>::max()) ?
                     std::numeric_limits<feature_value_t>::max() : boost::numeric_cast<feature_value_t>(parsed);
        }
    }

    // Figure out which features are binary, which are numeric
    numeric_features_.reserve(num_features_);
    for (unsigned f = 0; f < num_features_; ++f) {
        bool has_value_other_than_0_1 = false;
        for (unsigned s = 0; s < rowdata_.size(); ++s) {
            if (entry(s, f) > 1) {
                has_value_other_than_0_1 = true;
                break;
            }
        }
        numeric_features_.push_back(has_value_other_than_0_1);
    }
}


}

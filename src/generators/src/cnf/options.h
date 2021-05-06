
#pragma once

#include <string>
#include <vector>

namespace sltp::cnf {

struct Options {
    //! The path of the workspace where output files will be left
    std::string workspace;

    //! Whether we want to solve the maxsat problem or only generate the encoding
    bool solve;

    //! In the d2L encoding, whether we want to exploit the equivalence relation
    //! among transitions given by the feature pool
    bool use_equivalence_classes;

    //! The level of verbosity - higher value means more verbosity
    unsigned verbosity;

    //! In the d2L encoding, whether to post constraints to ensure distinguishability of goals
    bool distinguish_goals;

    //! The directory where the ASP encodings are
    std::string encodings_dir;

    //! The strategy to sample states when generating the encoding
    std::string sampling_strategy;

    //! An optional list with the a subset of features (feature IDs) that will be, if present, enforced as Selected;
    //! excluding the rest of features from the pool
    std::vector<unsigned> validate_features;

    //! The number of states initially sampled at random in the incremental approach
    unsigned initial_sample_size;

    //! The number of flaws to add to the set sample on each iteration of the incremental approach
    unsigned refinement_batch_size;

    //! The random seed
    unsigned seed;

    //! The number of features of the policy graph abstraction
    unsigned n_features;
};

Options parse_options(int argc, const char** argv);

}
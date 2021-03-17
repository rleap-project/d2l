
#include "factory.hxx"
#include <sltp/dl/elements.hxx>

#include <iostream>
#include <common/helpers.h>

namespace sltp::dl {

feature_sample_denotation_t compute_feature_sample_denotation(
        const Feature& feature, const Sample &sample, const Cache &cache, SampleDenotationProperties& properties) {

    const auto m = sample.num_states();

    feature_sample_denotation_t fd;
    fd.reserve(m);

    properties.denotation_is_bool = true;
    properties.denotation_is_constant = true;
    int previous_value = -1;

    for (unsigned sid = 0; sid < m; ++sid) {
        const State &state = sample.state(sid);
        assert(state.id() == sid);

        int value = feature.value(cache, sample, state);
        fd.push_back(value);

        properties.denotation_is_bool = properties.denotation_is_bool && (value < 2);
        properties.denotation_is_constant = (previous_value == -1)
                                            || (properties.denotation_is_constant && (previous_value == value));
        previous_value = value;
    }

    return fd;
}


inline feature_sample_denotation_t compute_feature_sample_denotation(
        const Feature& feature, const Sample &sample, const Cache &cache) {
    SampleDenotationProperties _;
    return compute_feature_sample_denotation(feature, sample, cache, _);
}


void Factory::log_all_concepts_and_features(
        const std::vector<const Concept*>& concepts,
        const Cache &cache, const Sample &sample, const std::string &workspace,
        bool print_denotations) {

    if (print_denotations) {
        std::cout << "Printing concept, role and feature denotations to " << workspace
                  << "/*-denotations.io.txt" << std::endl;
        // Print concept denotations
        std::string output(workspace + "/concept-denotations.io.txt");
        std::ofstream of(output);
        if (of.fail()) throw std::runtime_error("Could not open filename '" + output + "'");

        const auto m = sample.num_states();
        for (unsigned i = 0; i < m; ++i) {
            const State &state = sample.state(i);
            const auto& oidx = sample.instance(i).object_index();

            for (const Concept *c:concepts) {
                const state_denotation_t &denotation = cache.retrieveDLDenotation(*c, state, m);
                of << "s_" << i << "[" << c->str() << "] = {";
                bool need_comma = false;
                for (unsigned atom = 0; atom < denotation.size(); ++atom) {
                    if (denotation[atom]) {
                        if (need_comma) of << ", ";
                        of << oidx.right.at(atom);
                        need_comma = true;
                    }
                }
                of << "}" << std::endl;
            }
        }
        of.close();


        // Print role denotations
        output = workspace + "/role-denotations.io.txt";
        of = std::ofstream(output);
        if (of.fail()) throw std::runtime_error("Could not open filename '" + output + "'");

        for (unsigned i = 0; i < m; ++i) {
            const State &state = sample.state(i);
            const auto& oidx = sample.instance(i).object_index();
            unsigned n = sample.num_objects(i);


            for (const Role *r:roles_) {
                const state_denotation_t &denotation = cache.retrieveDLDenotation(*r, state, m);
                of << "s_" << i << "[" << r->str() << "] = {";
                bool need_comma = false;

                for (unsigned idx = 0; idx < denotation.size(); ++idx) {
                    if (denotation[idx]) {
                        if (need_comma) of << ", ";
                        unsigned o1 = idx / n;
                        unsigned o2 = idx % n;
                        of << "(" << oidx.right.at(o1) << ", " << oidx.right.at(o2) << ")";
                        need_comma = true;
                    }
                }
                of << "}" << std::endl;
            }
        }
        of.close();

        // Print feature denotations
        output = workspace + "/feature-denotations.io.txt";
        of = std::ofstream(output);
        if (of.fail()) throw std::runtime_error("Could not open filename '" + output + "'");

        for (unsigned i = 0; i < m; ++i) {
            const State &state = sample.state(i);

            for (const Feature *f:features_) {
                of << "s_" << i << "[" << f->as_str() << "] = " << f->value(cache, sample, state) << std::endl;
            }
        }
        of.close();
    }
    // Print all generated concepts
    std::string fname1 = workspace + "/serialized-concepts.io";
    std::ofstream of1 = std::ofstream(fname1);
    if( of1.fail() ) throw std::runtime_error("Could not open filename '" + fname1 + "'");

    // Print all generated features to be unserialized from the Python frontend
    std::string fname2 = workspace + "/serialized-features.io";
    std::ofstream of2 = std::ofstream(fname2);
    if( of2.fail() ) throw std::runtime_error("Could not open filename '" + fname2 + "'");

    std::cout << "Serializing all concepts and features to:\n\t" << fname1 << "\n\t" << fname2 << std::endl;
    for (const Concept* c:concepts) {
        of1 << c->str() << "\t" << c->complexity() << std::endl;
    }
    of1.close();

    for (const Feature* f:features_) {
        of2 << f->as_str() << "\t" << f->complexity() << std::endl;
    }
    of2.close();
}


void Factory::generate_features(
        const std::vector<const Concept*>& concepts,
        Cache &cache, const Sample &sample,
        const TransitionSample& transitions,
        const std::vector<const Concept*>& forced_goal_features)
{
    feature_cache_t seen_denotations;

    // Insert first the features that allow us to express the goal
    // goal_features will contain the indexes of those features
    goal_features_.clear();
    for (const auto *c:forced_goal_features) {
        if (attempt_cardinality_feature_insertion(c, cache, sample, transitions, seen_denotations, false)) {
            goal_features_.insert(features_.size()-1);
        }
    }
    std::cout << "A total of " << goal_features_.size() << " features were marked as goal-identifying" << std::endl;

    std::cout << "Generating cardinality features..." << std::endl;

    // create features that derive from nullary predicates
    // TODO Keep track of the denotation of nullary-atom features and prune them if necessary
    for (const auto& predicate:sample.predicates()) {
        if (predicate.arity() == 0) {
            features_.push_back(new NullaryAtomFeature(&predicate));
        }
    }

    // create boolean/numerical features from concepts
    for (const Concept* c:concepts) {
        attempt_cardinality_feature_insertion(c, cache, sample, transitions, seen_denotations, true);
    }

    // create comparison features here so that only cardinality features are used to build them
    generate_comparison_features(cache, sample, transitions, seen_denotations);

    // create distance features
    generate_distance_features(concepts, cache, sample, transitions, seen_denotations);

    // create conditional features from boolean conditions and numeric bodies
    generate_conditional_features(cache, sample, transitions, seen_denotations);

    print_feature_count();
}

bool Factory::attempt_cardinality_feature_insertion(
        const Concept* c,
        Cache &cache,
        const Sample &sample,
        const TransitionSample& transitions,
        feature_cache_t& seen_denotations,
        bool check_redundancy)
{
    // TODO We could compute the whole denotation with one single fetch of the underlying concept
    //      denotation. By doing as below, `compute_feature_sample_denotation` calls n times
    //      that same fetch, one per state in the sample.
    NumericalFeature nf(c);
    SampleDenotationProperties properties;
    auto fd = compute_feature_sample_denotation(nf, sample, cache, properties);

    if (prune_feature_denotation(nf, fd, properties, sample, transitions, seen_denotations, check_redundancy)) {
        return false;
    }

    const Feature *feature = properties.denotation_is_bool ?
                             static_cast<Feature*>(new BooleanFeature(c)) : static_cast<Feature*>(new NumericalFeature(c));

    features_.emplace_back(feature);
    seen_denotations.emplace(fd, feature);
    return true;
}

bool Factory::check_some_transition_pair_distinguished(const feature_sample_denotation_t &fsd, const Sample &sample,
                                                       const TransitionSample &transitions) {
// Make sure that the feature is useful to distinguish at least some pair of transitions
// coming from the same instance.
// Since the notion of distinguishability is transitive, we'll just check for one pair distinguished from the previous
// one
    int prev_instance_id = -1;
    bool feature_can_distinguish_some_transition = false;
    int last_sf = -1, last_sfprime = -1;
    for (auto s:transitions.all_alive()) {
        const State &state = sample.state(s);

        int sf = fsd[s];

        for (unsigned sprime:transitions.successors(s)) {
            int sfprime = fsd[sprime];

            if (last_sfprime > 0 && are_transitions_d1d2_distinguished(last_sf, last_sfprime, sf, sfprime)) {
                feature_can_distinguish_some_transition = true;
            }

            last_sfprime = sfprime;
            last_sf = sf;
        }

        if (prev_instance_id < 0) prev_instance_id = (int) state.instance_id();
        if (state.instance_id() != prev_instance_id) {
            last_sfprime = -1;
        }
    }
    return feature_can_distinguish_some_transition;
}

// TODO Ideally we'd want to unify this function with Factory::attempt_cardinality_feature_insertion, but
//  at we're not there yet
bool Factory::attempt_feature_insertion(
        const Feature* feature,
        unsigned bound,
        Cache &cache,
        const Sample &sample,
        const TransitionSample& transitions,
        feature_cache_t& seen_denotations,
        bool check_redundancy)
{
    if (feature->complexity() > bound) return false;

    SampleDenotationProperties properties;
    auto fd = compute_feature_sample_denotation(*feature, sample, cache, properties);

    if (prune_feature_denotation(
            *feature, fd, properties, sample, transitions, seen_denotations, check_redundancy)) {
        return false;
    }

    features_.emplace_back(feature);
    seen_denotations.emplace(fd, feature);

    return true;
}

bool Factory::prune_feature_denotation(
        const Feature& f,
        const feature_sample_denotation_t& fd,
        const SampleDenotationProperties& properties,
        const Sample &sample,
        const TransitionSample& transitions,
        feature_cache_t& seen_denotations,
        bool check_redundancy)
{
    // We want to determine:
    // - whether the feature is boolean or numeric (this is simply determined empirically: we consider it boolean
    //   if its value is always 0 or 1, and numeric otherwise),
    // - whether the full sample denotation of the feature coincides with some previous feature and hence we can prune
    //   it,
    // - whether the feature is not truly informative for our encodings, e.g. because it has the same variation over all
    //   transitions in the sample, or similar
    if (!check_redundancy) return false;

    if (properties.denotation_is_constant) return true;

    auto it = seen_denotations.find(fd);
    bool is_new = (it == seen_denotations.end());

    if (!is_new) {
//         std::cout << "REJECT (Redundant): " << f.fullstr() << std::endl;
        // Make sure that we don't prune a feature of lower complexity in favor of a feature of higher complexity
        // This should come for free, since features are ordered in increasing complexity
        if (it->second->complexity() > f.complexity()) {
            std::cout << sltp::utils::warning()
                      <<  "Feature " + f.as_str_with_complexity() + " has been pruned in favor of more complex "
                          + it->second->as_str_with_complexity() << std::endl;
        }

        return true;
    }

    if (!check_some_transition_pair_distinguished(fd, sample, transitions)) {
//         std::cout << "REJECT (NO DISTINCTION): " << f.fullstr() << std::endl;
        return true;
    }

//    std::cout << "ACCEPT: " << f.fullstr() << std::endl;
    return false;
}


const Predicate* Factory::get_role_predicate(const Role* r) {

    if (const auto *c = dynamic_cast<const PrimitiveRole*>(r)) {
        return c->predicate();

    } else if (const auto* c = dynamic_cast<const StarRole*>(r)) {
        return get_role_predicate(c->role());

    } else if (const auto *c = dynamic_cast<const PlusRole*>(r)) {
        return get_role_predicate(c->role());

    } else if (const auto *c = dynamic_cast<const RoleRestriction*>(r)) {
        return get_role_predicate(c->role());

    } else if (const auto *c = dynamic_cast<const InverseRole*>(r)) {
        return get_role_predicate(c->role());

    } else if (const auto *c = dynamic_cast<const RoleDifference*>(r)) {
        throw std::runtime_error("Unimplemented");
    }
    throw std::runtime_error("Unknown role type");
}


void Factory::generate_distance_features(
        const std::vector<const Concept*>& concepts, Cache &cache,
        const Sample &sample,
        const TransitionSample& transitions,
        feature_cache_t& seen_denotations) {
    if (options.dist_complexity_bound<=0) return;

    std::cout << "Generating distance features..." << std::endl;

    const auto m = sample.num_states();

    // Identify concepts with singleton denotation across all states: these are the candidates for start concepts
    std::vector<const Concept*> start_concepts;
    for (const Concept* c:concepts) {
        const sample_denotation_t& d = cache.find_sample_denotation(*c, m);
        bool singleton_denotations = true;
        for (unsigned j = 0; j < m; ++j) {
            if (d[j]->cardinality() != 1) {
                singleton_denotations = false;
                break;
            }
        }

        if (singleton_denotations) {
            start_concepts.push_back(c);
        }
    }

    // create role restrictions to be used in distance features
    std::vector<const Role*> role_restrictions(roles_);  // Start with all base roles
    for (const Role* r:roles_) {
        for (const Concept* c:concepts) {
            RoleRestriction role_restriction(r, c);
            if( role_restriction.complexity()+3 > options.dist_complexity_bound ) continue;
            const sample_denotation_t *d = role_restriction.denotation(cache, sample);

            if (!cache.contains(d)) {  // The role is not redundant
                role_restrictions.push_back(role_restriction.clone());
                cache.find_or_insert_sample_denotation(*d, role_restriction.id());
                //std::cout << "ACCEPT RR(sz=" << cache_for_role_restrictions.cache1().size() << "): "
                // + role_restriction.fullstr() << std::endl;
            } else {
                //std::cout << "PRUNE RR: " + role_restriction.str() << std::endl;
            }
            delete d;
        }
    }

    // create distance features
    int num_distance_features = 0;
    std::vector<const DistanceFeature*> candidates;

    for (const Concept* start:start_concepts) {
        for (const Concept* end:concepts) {
            if (start == end) continue;

            for (const Role* role:role_restrictions) {
                const auto* df = new DistanceFeature(start, end, role);
                if (df->complexity() > options.dist_complexity_bound) {
                    delete df;
                    continue;
                }

                candidates.push_back(df);
            }
        }
    }

    // Sort the candidate distance features along increasing complexity
    std::sort(std::begin(candidates), std::end(candidates),
              [](const DistanceFeature* f1, const DistanceFeature* f2) {
                  return f1->complexity() < f2->complexity();
              });

    for (const auto* df:candidates) {
        SampleDenotationProperties properties;
        const auto denotation = compute_feature_sample_denotation(*df, sample, cache, properties);

        if (!prune_feature_denotation(
                *df, denotation, properties, sample, transitions, seen_denotations, true)) {
            ++num_distance_features;
            features_.emplace_back(df);
            seen_denotations.emplace(denotation, features_.back());
        } else {
            delete df;
        }
    }
}


std::vector<const Concept*> Factory::consolidate_concepts() const {
    std::vector<const Concept*> all;
    for (const auto& layer:concepts_) all.insert(all.end(), layer.begin(), layer.end());

    std::stable_sort(std::begin(all), std::end(all), [](const Concept* c1, const Concept* c2) {
        return c1->complexity() < c2->complexity();
    });

    return all;
}


std::ostream& Factory::report_dl_data(std::ostream &os) const {
    unsigned nconcepts = 0;
    for (const auto& layer:concepts_) nconcepts += layer.size();
    auto nroles = roles_.size();
    os << "Total number of concepts / roles: " << nconcepts << "/" << nroles << std::endl;

    os << "Base concepts (sz=" << basis_concepts_.size() << "): ";
    for( int i = 0; i < int(basis_concepts_.size()); ++i ) {
        os << basis_concepts_[i]->str();
        if( 1 + i < int(basis_concepts_.size()) ) os << ", ";
    }
    os << std::endl;

    os << "Base roles (sz=" << basis_roles_.size() << "): ";
    for( int i = 0; i < int(basis_roles_.size()); ++i ) {
        os << basis_roles_[i]->str();
        if( 1 + i < int(basis_roles_.size()) ) os << ", ";
    }
    os << std::endl;

    os << "All Roles under complexity " << options.complexity_bound << " (sz=" << roles_.size() << "): ";
    for( int i = 0; i < int(roles_.size()); ++i ) {
        os << roles_[i]->fullstr();
        if( 1 + i < int(roles_.size()) ) os << ", ";
    }
    os << std::endl;

    os << "All concepts (by layer) under complexity " << options.complexity_bound << ": " << std::endl;
    for( unsigned layer = 0; layer < concepts_.size(); ++layer ) {
        os << "    Layer " << layer << " (sz=" << concepts_[layer].size() << "): ";
        for( unsigned i = 0; i < concepts_[layer].size(); ++i ) {
            os << concepts_[layer][i]->fullstr();
            if( 1 + i < concepts_[layer].size() ) os << ", ";
        }
        os << std::endl;
    }

    os << "Nullary-atom features: ";
    bool need_comma = false;
    for (const auto &feature : features_) {
        if( dynamic_cast<const NullaryAtomFeature*>(feature) ) {
            if( need_comma ) os << ", ";
            os << feature->as_str();
            need_comma = true;
        }
    }
    os << std::endl;

    os << "Boolean features: ";
    need_comma = false;
    for (const auto &feature : features_) {
        if( dynamic_cast<const BooleanFeature*>(feature) ) {
            if( need_comma ) os << ", ";
            os << feature->as_str();
            need_comma = true;
        }
    }
    os << std::endl;

    os << "Numerical features: ";
    need_comma = false;
    for (const auto &feature : features_) {
        if( dynamic_cast<const NumericalFeature*>(feature) ) {
            if( need_comma ) os << ", ";
            os << feature->as_str();
            need_comma = true;
        }
    }
    os << std::endl;

    os << "Distance features: ";
    need_comma = false;
    for (const auto &feature : features_) {
        if( dynamic_cast<const DistanceFeature*>(feature) ) {
            if( need_comma ) os << ", ";
            os << feature->as_str();
            need_comma = true;
        }
    }
    os << std::endl;

    return os;
}

void Factory::print_feature_count() const {
    unsigned num_nullary_features = 0, num_boolean_features = 0, num_numeric_features = 0,
            num_distance_features = 0, num_conditional_features = 0, num_comparison_features = 0;
    auto nf = features_.size();
    for (const auto *f:features_) {
        if (dynamic_cast<const NullaryAtomFeature*>(f)) num_nullary_features++;
        else if (dynamic_cast<const BooleanFeature*>(f)) num_boolean_features++;
        else if (dynamic_cast<const NumericalFeature*>(f)) num_numeric_features++;
        else if (dynamic_cast<const DistanceFeature*>(f)) num_distance_features++;
        else if (dynamic_cast<const ConditionalFeature*>(f)) num_conditional_features++;
        else if (dynamic_cast<const DifferenceFeature*>(f)) num_comparison_features++;
        else throw std::runtime_error("Unknown feature type");
    }

    assert(nf == num_nullary_features+num_boolean_features+num_numeric_features
                 +num_distance_features+num_conditional_features+num_comparison_features);
    std::cout << "FEATURES: #features=" << nf
              << ", #nullary="   << num_nullary_features
              << ", #boolean="   << num_boolean_features
              << ", #numerical=" << num_numeric_features
              << ", #distance="  << num_distance_features
              << ", #conditional="  << num_conditional_features
              << ", #comparison="  << num_comparison_features
              << std::endl;
}

void Factory::generate_conditional_features(
        Cache& cache,
        const Sample& sample,
        const TransitionSample& transitions,
        feature_cache_t& seen_denotations)
{
    if (options.cond_complexity_bound <= 0) return;

    std::cout << "Generating conditional features..." << std::endl;

    for (std::size_t i = 0, n = features_.size(); i < n; ++i) {
        const auto* cond = features_[i];
        if (!cond->is_boolean() || cond->complexity() + 1 + 1 > options.cond_complexity_bound) continue;

        for (std::size_t j = i + 1; j < n; ++j) {
            const auto* body = features_[j];
            if (body->is_boolean() ||
                cond->complexity() + body->complexity() + 1 > options.cond_complexity_bound)
                continue;

            const auto *feature = new ConditionalFeature(cond, body);
            if (!attempt_feature_insertion(
                    feature, options.cond_complexity_bound, cache, sample, transitions, seen_denotations, true)) {
                delete feature;
            }
        }
    }
}

void Factory::generate_comparison_features(
        Cache& cache,
        const Sample& sample,
        const TransitionSample& transitions,
        feature_cache_t& seen_denotations)
{
    if (!options.comparison_features) return;

    std::cout << "Generating comparison features..." << std::endl;

    // get the max index here, as we'll keep adding elements to the same `features_` vector:
    auto n = features_.size();

    for (std::size_t i = 0; i < n; ++i) {
        const auto* f_i = features_[i];
        if (f_i->is_boolean() || f_i->complexity() + 1 + 1 > options.complexity_bound) continue;

        for (std::size_t j = 0; j < n; ++j) {
            const auto* f_j = features_[j];
            if (i == j || f_j->is_boolean() || f_i->complexity() + f_j->complexity() + 1 > options.complexity_bound)
                continue;

            const auto *feature = new DifferenceFeature(f_i, f_j);
            if (!attempt_feature_insertion(
                    feature, options.complexity_bound, cache, sample, transitions, seen_denotations, true)) {
                delete feature;
            }
        }
    }
}

void Factory::generate_basis(const Sample &sample) {
    insert_basis(new UniversalConcept);
    insert_basis(new EmptyConcept);
    for( int i = 0; i < int(sample.num_predicates()); ++i ) {
        const Predicate *predicate = &sample.predicates()[i];
        if( predicate->arity() == 1 ) {
            insert_basis(new PrimitiveConcept(predicate));
        } else if( predicate->arity() == 2 ) {
            insert_basis(new PrimitiveRole(predicate));
        }
    }

    for (const auto& nominal:nominals_) {
        insert_basis(new NominalConcept(nominal));
    }
    std::cout << "BASIS: #concepts=" << basis_concepts_.size() << ", #roles=" << basis_roles_.size() << std::endl;
//        report_dl_data(std::cout);
}

int Factory::generate_roles(Cache &cache, const Sample &sample) {
    assert(roles_.empty());

    std::vector<const Role*> non_redundant_base_roles;

    // Insert the basis (i.e. primitive) roles as long as they are not redundant
    for (const auto *role : basis_roles_) {
        if (attempt_insertion(*role, cache, sample, roles_)) {
            non_redundant_base_roles.push_back(role);
        }
    }

    // Now insert a few compounds based on those base roles that are not redundant
    for (const auto *role:non_redundant_base_roles) {

        // Create Inverse(R) role from the primitive role
        attempt_insertion(InverseRole(role), cache, sample, roles_);

        // Create Plus(R) role from the primitive role
        PlusRole p_role(role);
        if (attempt_insertion(p_role, cache, sample, roles_)) {
            // Create Inverse(Plus(R)) only if Plus(R) is NOT redundant
            attempt_insertion(InverseRole(p_role.clone()), cache, sample, roles_);
        }
        // Create Star(R) roles from the primitive roles !!! NOTE ATM we deactivate Star roles
        // attempt_role_insertion(StarRole(role), cache, sample);
    }
    std::cout << "ROLES: #roles=" << roles_.size() << std::endl;
    return roles_.size();
}

std::vector<const Concept*> Factory::generate_concepts(Cache &cache, const Sample &sample, const std::clock_t& start_time) {
    std::size_t num_concepts = 0;
    bool some_new_concepts = true;
    bool timeout_reached = false;

    if (!concepts_.empty()) throw std::runtime_error("Don't invoke Factory::generate_concepts more than once");

    // Start by processing the basis concepts
    concepts_.emplace_back();
    for (const auto* concept : basis_concepts_) attempt_insertion(*concept, cache, sample, concepts_.back());

    for( int iteration = 0; some_new_concepts && !timeout_reached; ++iteration ) {
        std::cout << "Start of iteration " << iteration
                  << ": #total-concepts=" << num_concepts
                  << ", #concepts-in-last-layer=" << (concepts_.empty() ? 0 : concepts_.back().size())
                  << std::endl;

        std::size_t num_concepts_before_step = num_concepts;

        // Let the fun begin:
        auto num_pruned_concepts = advance_step(cache, sample, start_time);
        timeout_reached = num_pruned_concepts < 0;

        num_concepts += concepts_.empty() ? 0 : concepts_.back().size();
        some_new_concepts = num_concepts > num_concepts_before_step;

        std::cout << "Result of iteration " << iteration
                  << ": #concepts-in-layer=" << concepts_.back().size()
                  << ", #pruned-concepts=" << num_pruned_concepts
                  << std::endl;
//            report_dl_data(std::cout);
    }

    if (concepts_.back().empty()) {  // i.e. we stopped because last iteration was already a fixpoint
        concepts_.pop_back();
    }

    // Important to use this vector from here on, as it is sorted by complexity
    auto all_concepts = consolidate_concepts();
    assert(all_concepts.size() == num_concepts);

    std::cout << "dl::Factory: #concepts-final=" << num_concepts << std::endl;
    return all_concepts;
}

int Factory::advance_step(Cache &cache, const Sample &sample, const std::clock_t& start_time) {
    int num_pruned_concepts = 0;

    // Classify concepts and roles by their complexity - we will use this to minimize complexity checks later
    std::vector<std::vector<const Role*>> roles_by_complexity(options.complexity_bound+1);
    std::vector<std::vector<const Concept*>> concepts_in_last_layer_by_complexity(options.complexity_bound+1);
    std::vector<std::vector<const Concept*>> concepts_in_previous_layers_by_complexity(options.complexity_bound+1);

    for (const auto* r:roles_) roles_by_complexity.at(r->complexity()).push_back(r);  // TODO Do this just once
    for (const auto* c:concepts_.back()) concepts_in_last_layer_by_complexity.at(c->complexity()).push_back(c);

    for( unsigned layer = 0; layer < concepts_.size()-1; ++layer ) {
        for (const auto* c:concepts_[layer]) {
            concepts_in_previous_layers_by_complexity.at(c->complexity()).push_back(c);
        }
    }

    // create new concept layer
    bool is_first_non_basis_iteration = (concepts_.size() == 1);
    concepts_.emplace_back();

    if (is_first_non_basis_iteration) {
        // Let's create equal concepts for those pairs of roles (whose number is already fixed at this point)
        // such that both arise from same predicate and one of them is the "goal version" of the other, e.g.
        // on and on_g, in blocksworld.
        for (const auto* r1:roles_) {
            for (const auto* r2:roles_) {
                if (dynamic_cast<const RoleDifference*>(r1) || dynamic_cast<const RoleDifference*>(r2)) {
                    // R=R' Makes no sense when R or R' are already a role difference
                    continue;
                }
                auto name1 = get_role_predicate(r1)->name();
                auto name2 = get_role_predicate(r2)->name();

                // A hacky way to allow only equal concepts R=R' where R and R' come from the same predicate
                // and the first one is a goal role
                if (name1.substr(name1.size()-2) != "_g") continue;
                if (name1.substr(0, name1.size()-2) != name2) continue;

                EqualConcept eq_concept(r1, r2);
                // Force the complexity of R=R' to be 1, if both roles are of the same type
                // This is currently disabled, as the concept-pruning strategy doesn't work as expected when
                // this is enabled - sometimes concepts with lower complexity get pruned in favor of others
                // with higher complexity
                // if (typeid(*r1) == typeid(*r2)) eq_concept.force_complexity(1);

                num_pruned_concepts += !attempt_insertion(eq_concept, cache, sample, concepts_.back());

                if (check_timeout(start_time)) return -1;
            }
        }
    }

    for (int k = 0; k <= (options.complexity_bound-1); ++k) {
        for (const auto* concept:concepts_in_last_layer_by_complexity[k]) {

            // Negate concepts in the last layer, only if they are not already negations
            if (!dynamic_cast<const NotConcept*>(concept)) {
                num_pruned_concepts += !attempt_insertion(NotConcept(concept), cache, sample, concepts_.back());
            }


            // generate exist and forall combining a role with a concept in the last layer
            for (int k2 = 0; k2 <= (options.complexity_bound-k-1); ++k2) {
                for (const auto *role:roles_by_complexity[k2]) {
                    num_pruned_concepts += !attempt_insertion(ExistsConcept(concept, role), cache, sample, concepts_.back());
                    num_pruned_concepts += !attempt_insertion(ForallConcept(concept, role), cache, sample, concepts_.back());

                    if (check_timeout(start_time)) return -1;
                }
            }
            if (check_timeout(start_time)) return -1;
        }
    }


    // generate conjunctions of (a) a concept of the last layer with a concept in some previous layer, and
    // (b) two concepts of the last layer, avoiding symmetries
    for (int k = 0; k <= (options.complexity_bound-1); ++k) {
        for (unsigned i_k = 0; i_k < concepts_in_last_layer_by_complexity[k].size(); ++i_k) {
            const auto& concept1 = concepts_in_last_layer_by_complexity[k][i_k];

            // (a) a concept of the last layer with a concept in some previous layer
            for (int k2 = 0; k2 <= (options.complexity_bound-k); ++k2) {
                for (const auto *concept2:concepts_in_previous_layers_by_complexity[k2]) {
                    num_pruned_concepts += !attempt_insertion(AndConcept(concept1, concept2), cache, sample, concepts_.back());

                    if (options.generate_or_concepts) {
                        num_pruned_concepts += !attempt_insertion(OrConcept(concept1, concept2), cache, sample, concepts_.back());
                    }

                    if (check_timeout(start_time)) return -1;
                }
            }


            // (b) two concepts of the last layer, avoiding symmetries
            for (int k2 = k; k2 <= (options.complexity_bound-k); ++k2) {
                unsigned start_at = (k == k2) ? i_k+1: 0; // Break symmetries within same complexity bucket
                for (unsigned i_k2 = start_at; i_k2 < concepts_in_last_layer_by_complexity[k2].size(); ++i_k2) {
                    const auto& concept2 = concepts_in_last_layer_by_complexity[k2][i_k2];
                    num_pruned_concepts += !attempt_insertion(AndConcept(concept1, concept2), cache, sample, concepts_.back());

                    if (options.generate_or_concepts) {
                        num_pruned_concepts += !attempt_insertion(OrConcept(concept1, concept2), cache, sample, concepts_.back());
                    }

                    if (check_timeout(start_time)) return -1;
                }
            }
        }
    }
    return num_pruned_concepts;
}


std::vector<const Concept*> Factory::generate_goal_concepts_and_roles(Cache &cache, const Sample &sample) {
    std::vector<const Concept*> goal_concepts;

    const Sample::PredicateIndex& predicate_idx = sample.predicate_index();
    for (const predicate_id_t& pid:sample.goal_predicates()) {
        const Predicate& pred = sample.predicate(pid);
        unsigned arity = pred.arity();
        const auto& name = pred.name();
        auto name_n = name.size();
        assert(name[name_n-2] == '_' && name[name_n-1] == 'g');  // Could try something more robust, but ATM this is OK
        auto nongoal_name = name.substr(0, name_n-2);
        const Predicate& nongoal_pred = sample.predicate(predicate_idx.at(nongoal_name));
        if (arity == 0) {
            // No need to do anything here, as the corresponding NullaryAtomFeature will always be generated
            // with complexity 1
            assert (0); // This will need some refactoring
//                goal_features.push_back(new NullaryAtomFeature(&pred));

        } else if (arity == 1) {
            // Force into the basis a concept "p_g AND Not p"
            auto *c = new PrimitiveConcept(&nongoal_pred);
            auto *not_c = new NotConcept(c);
            auto *c_g = new PrimitiveConcept(&pred);
            auto *and_c = new AndConcept(c_g, not_c);

            // Insert the denotations into the cache
//                for (const auto elem:std::vector<const Concept*>({and_c})) {
            for (const auto* elem:std::vector<const Concept*>({c, not_c, c_g, and_c})) {
                const sample_denotation_t *d = elem->denotation(cache, sample);
                cache.find_or_insert_sample_denotation(*d, elem->id());
            }
//                for (const auto elem:std::vector<const Concept*>({c, not_c, c_g})) {
//                    cache.remove_sample_denotation(elem->str());
//                }


            and_c->force_complexity(1);
            for (const auto* elem:std::vector<const Concept*>({c, not_c, c_g, and_c})) {
                insert_basis(elem);
            }

//                goal_features.push_back(new NumericalFeature(and_c));
            goal_concepts.push_back(and_c);

        } else if (arity == 2) {
            // Force into the basis a new RoleDifference(r_g, r)
            auto *r = new PrimitiveRole(&nongoal_pred);
            auto *r_g = new PrimitiveRole(&pred);
            auto *r_diff = new RoleDifference(r_g, r);

            // Insert the denotations into the cache
//                for (const auto elem:std::vector<const Role*>({r_diff})) {
//                for (const auto elem:std::vector<const Role*>({r, r_g, r_diff})) {
//                    const sample_denotation_t *d = elem->denotation(cache, sample, false);
//                    cache.find_or_insert_sample_denotation(*d, elem->str());
//                }
//                for (const auto elem:std::vector<const Role*>({r, r_g})) {
//                    cache.remove_sample_denotation(elem->str());
//                }

            r_diff->force_complexity(1);
            for (const auto* elem:std::vector<const Role*>({r, r_g, r_diff})) {
                insert_basis(elem);
            }

            auto *ex_c = new ExistsConcept(new UniversalConcept, r_diff); // memleak
//                goal_features.push_back(new NumericalFeature(ex_c));
            goal_concepts.push_back(ex_c);
        }
    }
    return goal_concepts;
}

void Factory::output_feature_matrix(std::ostream &os, const Cache &cache, const Sample &sample, const sltp::TransitionSample& transitions) const {
    auto num_features = features_.size();
    unsigned nfeatures_to_print = options.print_hstar ? num_features+1 : num_features;

    os << ";; Feature matrix of " << nfeatures_to_print << " features times " << sample.num_states() << " states."
       << " Format: all space-separated values. Line 0: This header line. Line 1: Feature names; Line 2: Feature complexities; Line i, 3<=i<=n: Feature valuation in state #i"
       << std::endl;

    // Line #1: feature names
    for (unsigned i = 0; i < num_features; ++i) {
        const Feature &feature = *features_[i];
        os << feature.as_str();
        if (1 + i < num_features) os << " ";
    }
    if (options.print_hstar) os << " hstar";
    os << std::endl;

    // Line #2: feature complexities
    for (unsigned i = 0; i < num_features; ++i) {
        const Feature &feature = *features_[i];
        os << feature.complexity();
        if (1 + i < num_features) os << " ";
    }
    if (options.print_hstar) os << " -1";
    os << std::endl;


    // Next: one line per state with the numeric denotation of all features
    for (unsigned i = 0; i < sample.num_states(); ++i) {
        const State &state = sample.state(i);

        for (unsigned j = 0; j < num_features; ++j) {
            const Feature &feature = *features_[j];
            auto val = feature.value(cache, sample, state);
            if (options.infty_feat_as_zero && val == Feature::infty) val = 0;
            os << val;
            if (1 + j < num_features) os << " ";
        }

        if (options.print_hstar) os << " " << transitions.vstar(i);
        os << std::endl;
    }
}


template <typename T1, typename T2>
bool Factory::attempt_insertion(const T1& elem, Cache &cache, const Sample &sample, std::vector<const T2*>& container) const {
    if (elem.complexity() > options.complexity_bound) {
//            std::cout << elem.str() << " superfluous because complexity " << elem.complexity() << ">" << options.complexity_bound << std::endl;
        return false;
    }

    const sample_denotation_t *d = elem.denotation(cache, sample);

    const auto& index = cache.cache1();
    auto it = index.find(d);
    if (it != index.end()) {
        // There is in the index some other concept/role with same sample denotation,
        // hence we consider this one redundant
//        std::cout << elem.str() << " redundant with concept with ID: " << it->second << std::endl;
        delete d;
        return false;

    } else {
        container.push_back(elem.clone());
        assert (container.back()->id() == elem.id());
//        std::cout << elem.fullstr() << " inserted" << std::endl;
        cache.find_or_insert_sample_denotation(*d, elem.id());
        delete d;
        return true;
    }
}

} // namespaces

#ifndef FEATURES_HXX
#define FEATURES_HXX

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cassert>
#include <utility>

#include <sltp/utils.hxx>

namespace SLTP {

namespace DL {

using object_id = unsigned;
using predicate_id = unsigned;
using atom_id = unsigned;
using state_id = unsigned;

class GroundedPredicate;

// A state denotation for concept C is a subset of objects
// implemented as a bitmap (ie. vector<bool>). These are
// cached so that at most one copy of a bitmap exists.
struct state_denotation_t : public std::vector<bool> {
    state_denotation_t(size_t n, bool value) : std::vector<bool>(n, value) { }
    size_t cardinality() const {
        size_t n = 0;
        for( size_t i = 0; i < size(); ++i )
            n += (*this)[i];
        return n;
    }
};

// A (full) sample denotation for concept C is a vector of
// state denotations, one per each state in the sample.
// Since we cache state denotations, a sample denotation
// is implemented as a vector of pointers to bitmaps.
struct sample_denotation_t : public std::vector<const state_denotation_t*> { };

// We cache sample and state denotations. The latter are cached
// into a simple hash (i.e. unordered_set). The former are cached
// using two hash maps (i.e. unordered_map): one that maps sample
// denotations to concept names, and the other that maps concept
// names to sample denotations.
//
// We also cache grounded predicates that are used to 
// represent states in the sample.
class Cache {
  public:
    struct cache_support_t {
        // cache for sample denotations
        bool operator()(const sample_denotation_t *d1, const sample_denotation_t *d2) const {
            assert((d1 != nullptr) && (d2 != nullptr));
            return *d1 == *d2;
        }
        size_t operator()(const sample_denotation_t *obj) const {
            assert(obj != nullptr);
            size_t hash = (*this)((*obj)[0]);
            for( int i = 0; i < int(obj->size()); ++i )
                hash = hash ^ (*this)((*obj)[i]);
            return hash;
        }

        // cache for state denotations
        bool operator()(const state_denotation_t *sd1, const state_denotation_t *sd2) const {
            assert((sd1 != nullptr) && (sd2 != nullptr));
            return *sd1 == *sd2;
        }
        size_t operator()(const state_denotation_t *obj) const {
            assert(obj != nullptr);
            std::hash<std::vector<bool> > hasher;
            return hasher(*static_cast<const std::vector<bool>*>(obj));
        }
    };

    typedef typename std::unordered_map<const sample_denotation_t*, std::string, cache_support_t, cache_support_t> cache1_t;
    typedef typename std::unordered_map<std::string, const sample_denotation_t*> cache2_t;
    typedef typename std::unordered_set<const state_denotation_t*, cache_support_t, cache_support_t> cache3_t;
    typedef typename std::unordered_map<std::string, const GroundedPredicate*> cache4_t;

  protected:
    cache1_t cache1_;
    cache2_t cache2_;
    cache3_t cache3_;
    cache4_t cache4_;

  public:
    Cache() { }
    ~Cache() { }

    // cache1: (full) sample denotations for concepts
    //
    // We maintain a hash of (full) sample denotations so that one copy
    // exists for each such denotations. This hash is implemented with
    // two tables, one that provides mappings from sample denotations
    // to concept names, and the other is the inverse
    //
    // sample denotation -> concept name
    const sample_denotation_t* find_sample_denotation(const sample_denotation_t &d) const {
        cache1_t::const_iterator it = cache1_.find(&d);
        return it == cache1_.end() ? nullptr : it->first;
    }
    const sample_denotation_t* find_or_insert_sample_denotation(const sample_denotation_t &d, const std::string &name) {
        cache1_t::const_iterator it = cache1_.find(&d);
        if( it == cache1_.end() ) {
            const sample_denotation_t *nd = new sample_denotation_t(d);
            cache1_.emplace(nd, name);
            cache2_.emplace(name, nd);
            return nd;
        } else {
            return it->first;
        }
    }

    // cache2: (full) sample denotations for concepts
    //
    // concept name -> sample denotation
    const sample_denotation_t* find_sample_denotation(const std::string name) const {
        cache2_t::const_iterator it = cache2_.find(name);
        return it == cache2_.end() ? nullptr : it->second;
    }

    // cache3: state denotations (bit vectors)
    //
    // We maintain a hash of these denotations such as to keep just one
    // copy of each such bitmap. A sample denotation is a vector of
    // state denotations. The same bitmap may appear associated with
    // different states in the same sample, or across sample denotations
    // for different concepts
    //
    // state denotation -> pointer
    const state_denotation_t* find_state_denotation(const state_denotation_t &sd) const {
        cache3_t::const_iterator it = cache3_.find(&sd);
        return it == cache3_.end() ? nullptr : *it;
    }

    const state_denotation_t* find_or_insert_state_denotation(const state_denotation_t &sd) {
        cache3_t::const_iterator it = cache3_.find(&sd);
        if( it == cache3_.end() ) {
            const state_denotation_t *nsd = new state_denotation_t(sd);
            cache3_.insert(nsd);
            return nsd;
        } else {
            return *it;
        }
    }

    // cache4: grounded predicates
    //
    // name -> grounded predicate
    const GroundedPredicate* find_grounded_predicate(const std::string &name) const {
        cache4_t::const_iterator it = cache4_.find(name);
        return it == cache4_.end() ? nullptr : it->second;
    }
    const GroundedPredicate* find_or_insert_grounded_predicate(const GroundedPredicate &gp);
};

// We represent states as subets of grounded predicates.
// A grounded predicate is a predicate and a vector of
// objects to ground the predicates.
class Object {
  protected:
    const object_id id_;
    const std::string name_;

  public:
    Object(unsigned id, std::string name) : id_(id), name_(std::move(name)) { }
    int id() const {
        return id_;
    }
    const std::string& name() const {
        return name_;
    }
    const std::string& as_str() const {
        return name_;
    }
    friend std::ostream& operator<<(std::ostream &os, const Object &obj) {
        return os << obj.as_str() << std::flush;
    }
};

struct Predicate {
    const predicate_id id_;
    const std::string name_;
    const unsigned arity_;
    Predicate(unsigned id, std::string name, unsigned arity) : id_(id), name_(std::move(name)), arity_(arity) { }
    unsigned arity() const {
        return arity_;
    }
    std::string as_str(const std::vector<object_id> *objects) const {
        std::string str = name_ + "(";
        if( objects == nullptr ) {
            for( int i = 0; i < arity_; ++i ) {
                str += std::string("x") + std::to_string(1 + i);
                if( 1 + i < arity_ ) str += ",";
            }
        } else {
            assert(objects->size() == arity_);
            for( int i = 0; i < arity_; ++i ) {
//                str += (*objects)[i]->as_str();
                str += std::to_string((*objects)[i]);
                if( 1 + i < arity_ ) str += ",";
            }
        }
        return str + ")";
    }

    predicate_id id() const { return id_; }

    friend std::ostream& operator<<(std::ostream &os, const Predicate &pred) {
        return os << pred.as_str(nullptr) << std::flush;
    }
};

class GroundedPredicate {
  protected:
    const predicate_id predicate_;
    const std::vector<object_id> objects_;

  public:
    GroundedPredicate(const predicate_id& predicate, std::vector<object_id> objects)
      : predicate_(predicate), objects_(std::move(objects)) {
    }
    GroundedPredicate(const predicate_id& predicate, std::vector<object_id> &&objects)
      : predicate_(predicate), objects_(std::move(objects)) {
    }

    predicate_id pred_id() const { return predicate_; }
    const std::vector<object_id>& objects() const { return objects_; }

    //! Return the i-th object of the current atom
    object_id object(unsigned order) const { return objects_.at(order); }

    bool is_instance(const Predicate& predicate) const {
        return predicate_ == predicate.id();
    }

    std::vector<unsigned> data() const {
        std::vector<unsigned> res{predicate_};
        for (const auto x:objects_) res.push_back(x);
        return res;
    }

//    std::string as_str() const {
//        return predicate_.as_str(&objects_);
//    }
//    friend std::ostream& operator<<(std::ostream &os, const GroundedPredicate &pred) {
//        return os << pred.as_str() << std::flush;
//    }
};

// A state is a collections of atoms (i.e. GroundedPredicates)
class State {
  protected:
    const state_id id_;
    std::vector<atom_id> atoms_;

  public:
    State(unsigned id, const std::vector<atom_id>& atoms) : id_(id), atoms_(atoms) { }
    unsigned id() const {
        return id_;
    }
    const std::vector<atom_id>& atoms() const {
        return atoms_;
    }
};

// A sample is a bunch of states and transitions among them. The
// sample contains the predicates used in the states, the objects,
// and the grounded predicates (i.e. atoms).
class Sample {
public:
    using ObjectIndex = std::unordered_map<std::string, object_id>;
    using PredicateIndex = std::unordered_map<std::string, predicate_id>;

    //! Map from <pred_id, oid_1, ..., oid_n> to atom ID
    using AtomIndex = std::unordered_map<std::vector<unsigned>,
            atom_id, utils::container_hash<std::vector<unsigned>>>;

  protected:
    const std::string name_;
    const std::vector<Object> objects_;
    const std::vector<Predicate> predicates_;
    const std::vector<GroundedPredicate> grounded_predicates_;
    const std::vector<State> states_;

    //! A mapping from object names to their ID in the sample
    ObjectIndex oidx_;

    //! A mapping from predicate names to their ID in the sample
    PredicateIndex pidx_;

    //! A mapping from <predicate name, obj_name, ..., obj_name> to the ID of the corresponding GroundPredicate
    AtomIndex aidx_;

    Sample(std::string name,
           std::vector<Object> objects,
           std::vector<Predicate> predicates,
           std::vector<GroundedPredicate> grounded_predicates,
           std::vector<State> states,
           ObjectIndex oidx,
           PredicateIndex pidx,
           AtomIndex aidx) :
        name_(std::move(name)),
        objects_(std::move(objects)),
        predicates_(std::move(predicates)),
        grounded_predicates_(std::move(grounded_predicates)),
        states_(std::move(states)),
        oidx_(std::move(oidx)),
        pidx_(std::move(pidx)),
        aidx_(std::move(aidx))
    { }

public:
    ~Sample() = default;

    const std::string& name() const {
        return name_;
    }
    std::size_t num_objects() const {
        return objects_.size();
    }
    std::size_t num_predicates() const {
        return predicates_.size();
    }
    std::size_t num_grounded_predicates() const {
        return grounded_predicates_.size();
    }
    std::size_t num_states() const {
        return states_.size();
    }

    const std::vector<Predicate>& predicates() const {
        return predicates_;
    }
    const std::vector<State>& states() const {
        return states_;
    }

    const State& state(unsigned id) const { return states_.at(id); }

    const GroundedPredicate& atom(unsigned id) const { return grounded_predicates_.at(id); }

    bool is_instance(atom_id atom, const Predicate& predicate) const {
        return grounded_predicates_.at(atom).is_instance(predicate);
    }

    //! Factory method - reads sample from serialized data
    static Sample read(std::istream &is);
};

struct Denotation {
    enum class denotation_t : bool { concept_denotation_t, role_denotation_t };
    denotation_t type_;
    std::vector<bool> values_;
    Denotation(denotation_t type, size_t dimension)
      : type_(type),
        values_(std::vector<bool>(dimension, false)) {
    }
};

// Denotation matrix is just a matrix of DL Denotations, each state being a row, each concept / role a column
class DenotationMatrix {
  protected:
    std::vector<std::vector<Denotation> > data_;
};

class Model {
};

class Base {
  protected:
    // unsigned id_;
    unsigned complexity_;

  public:
    explicit Base(unsigned complexity) : complexity_(complexity) { }
    // const unsigned id() const { return id_; }
    const unsigned complexity() const { return complexity_; }

    //virtual void denotation(Denotation &d) const = 0;
    virtual const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const = 0;
    virtual const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const = 0;
    virtual std::string as_str() const = 0;

    friend std::ostream& operator<<(std::ostream &os, const Base &base) {
        return os << base.as_str() << std::flush;
    }
};

class Concept : public Base {
  public:
    explicit Concept(unsigned complexity) : Base(complexity) { }
    virtual ~Concept() { }
};

class Role : public Base {
  public:
    explicit Role(unsigned complexity) : Base(complexity) { }
    virtual ~Role() { }
};

class PrimitiveConcept : public Concept {
  protected:
    const Predicate *predicate_;

  public:
    PrimitiveConcept(const Predicate *predicate) : Concept(1), predicate_(predicate) { }
    virtual ~PrimitiveConcept() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            sample_denotation_t d;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const State &s = sample.state(i);
                d.emplace_back(denotation(cache, sample, s));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(d, as_str()) : new sample_denotation_t(d);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        state_denotation_t sd(sample.num_objects(), false);
        for (atom_id id : state.atoms()) {
            const GroundedPredicate& atom = sample.atom(id);
            if (atom.is_instance(*predicate_)) {
                assert(atom.objects().size() == 1);
                object_id index = atom.object(0);
                assert(index < sample.num_objects());
                sd[index] = true;
            }
        }
        return cache.find_or_insert_state_denotation(sd);
    }
    std::string as_str() const override {
        return predicate_->name_;
    }
};

class UniversalConcept : public Concept {
  public:
    UniversalConcept() : Concept(1) { }
    virtual ~UniversalConcept() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            state_denotation_t sd(sample.num_objects(), true);
            const state_denotation_t *cached_sd = cache.find_or_insert_state_denotation(sd);

            sample_denotation_t nd;
            for( unsigned i = 0; i < sample.num_states(); ++i )
                nd.emplace_back(cached_sd);
            return use_cache ? cache.find_or_insert_sample_denotation(nd, as_str()) : new sample_denotation_t(nd);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: UniversalConcept::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }
    std::string as_str() const override {
        return "<universal>";
    }
};

class AndConcept : public Concept {
  protected:
    const Concept *concept1_;
    const Concept *concept2_;

  public:
    AndConcept(const Concept *concept1, const Concept *concept2)
      : Concept(1 + concept1->complexity() + concept2->complexity()),
        concept1_(concept1),
        concept2_(concept2) {
    }
    virtual ~AndConcept() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            const sample_denotation_t *d1 = cache.find_sample_denotation(concept1_->as_str());
            assert((d1 != nullptr) && (d1->size() == sample.num_states()));
            const sample_denotation_t *d2 = cache.find_sample_denotation(concept2_->as_str());
            assert((d2 != nullptr) && (d2->size() == sample.num_states()));

            sample_denotation_t nd;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const state_denotation_t *sd1 = (*d1)[i];
                assert((sd1 != nullptr) && (sd1->size() == sample.num_objects()));
                const state_denotation_t *sd2 = (*d2)[i];
                assert((sd2 != nullptr) && (sd2->size() == sample.num_objects()));

                state_denotation_t nsd(sample.num_objects(), false);
                for( int j = 0; j < sample.num_objects(); ++j )
                    nsd[j] = (*sd1)[j] && (*sd2)[j];
                nd.emplace_back(cache.find_or_insert_state_denotation(nsd));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nd, as_str()) : new sample_denotation_t(nd);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: AndConcept::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }

    std::string as_str() const override {
        return std::string("And[") +  concept1_->as_str() + "," + concept2_->as_str() + "]";
    }
};

class NotConcept : public Concept {
  protected:
    const Concept *concept_;

  public:
    NotConcept(const Concept *concept)
      : Concept(1 + concept->complexity()),
        concept_(concept) {
    }
    virtual ~NotConcept() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            const sample_denotation_t *d = cache.find_sample_denotation(concept_->as_str());
            assert((d != nullptr) && (d->size() == sample.num_states()));

            sample_denotation_t nd;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const state_denotation_t *sd = (*d)[i];
                assert((sd != nullptr) && (sd->size() == sample.num_objects()));

                state_denotation_t nsd(sample.num_objects(), false);
                for( int j = 0; j < sample.num_objects(); ++j )
                    nsd[j] = !(*sd)[j];
                nd.emplace_back(cache.find_or_insert_state_denotation(nsd));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nd, as_str()) : new sample_denotation_t(nd);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: NotConcept::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }

    std::string as_str() const override {
        return std::string("Not[") + concept_->as_str() + "]";
    }
};

class ExistsConcept : public Concept {
  protected:
    const Concept *concept_;
    const Role *role_;

  public:
    ExistsConcept(const Concept *concept, const Role *role)
      : Concept(1 + concept->complexity() + role->complexity()),
        concept_(concept),
        role_(role) {
    }
    virtual ~ExistsConcept() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            const sample_denotation_t *d = cache.find_sample_denotation(concept_->as_str());
            assert((d != nullptr) && (d->size() == sample.num_states()));
            const sample_denotation_t *r = cache.find_sample_denotation(role_->as_str());
            assert((r != nullptr) && (r->size() == sample.num_states()));

            sample_denotation_t nd;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const state_denotation_t *sd = (*d)[i];
                assert((sd != nullptr) && (sd->size() == sample.num_objects()));
                const state_denotation_t *sr = (*r)[i];
                assert((sr != nullptr) && (sr->size() == sample.num_objects() * sample.num_objects()));

                state_denotation_t nsd(sample.num_objects(), false);
                for( int j = 0; j < sample.num_objects(); ++j ) {
                    if( (*sd)[j] ) {
                        for( int k = 0; !nsd[j] && (k < sample.num_objects()); ++k ) {
                            int index = j * sample.num_objects() + k;
                            nsd[j] = (*(*r)[i])[index];
                        }
                    }
                }
                nd.emplace_back(cache.find_or_insert_state_denotation(nsd));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nd, as_str()) : new sample_denotation_t(nd);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: ExistsConcept::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }

    std::string as_str() const override {
        return std::string("Exists[") + concept_->as_str() + "," + role_->as_str() + "]";
    }
};

class ForallConcept : public Concept {
  protected:
    const Concept *concept_;
    const Role *role_;

  public:
    ForallConcept(const Concept *concept, const Role *role)
      : Concept(1 + concept->complexity() + role->complexity()),
        concept_(concept),
        role_(role) {
    }
    virtual ~ForallConcept() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            const sample_denotation_t *d = cache.find_sample_denotation(concept_->as_str());
            assert((d != nullptr) && (d->size() == sample.num_states()));
            const sample_denotation_t *r = cache.find_sample_denotation(role_->as_str());
            assert((r != nullptr) && (r->size() == sample.num_states()));

            sample_denotation_t nd;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const state_denotation_t *sd = (*d)[i];
                assert((sd != nullptr) && (sd->size() == sample.num_objects()));
                const state_denotation_t *sr = (*r)[i];
                assert((sr != nullptr) && (sr->size() == sample.num_objects() * sample.num_objects()));

                state_denotation_t nsd(sample.num_objects(), true);
                for( int j = 0; j < sample.num_objects(); ++j ) {
                    if( (*sd)[j] ) {
                        for( int k = 0; nsd[j] && (k < sample.num_objects()); ++k ) {
                            int index = j * sample.num_objects() + k;
                            nsd[j] = (*(*r)[i])[index];
                        }
                    }
                }
                nd.emplace_back(cache.find_or_insert_state_denotation(nsd));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nd, as_str()) : new sample_denotation_t(nd);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: ForallConcept::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }

    std::string as_str() const override {
        return std::string("Forall[") + concept_->as_str() + "," + role_->as_str() + "]";
    }
};

class PrimitiveRole : public Role {
  protected:
    const Predicate *predicate_;

  public:
    PrimitiveRole(const Predicate *predicate) : Role(1), predicate_(predicate) { }
    virtual ~PrimitiveRole() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            sample_denotation_t nr;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const State &s = sample.state(i);
                nr.emplace_back(denotation(cache, sample, s));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nr, as_str()) : new sample_denotation_t(nr);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        state_denotation_t sr(sample.num_objects() * sample.num_objects(), false);
        for (atom_id id : state.atoms()) {
            const GroundedPredicate& atom = sample.atom(id);
            if (atom.is_instance(*predicate_)) {
                assert(atom.objects().size() == 2);
                unsigned index = atom.object(0) * sample.num_objects() + atom.object(1);
                assert(index < sample.num_objects() * sample.num_objects());
                sr[index] = true;
            }
        }
        return cache.find_or_insert_state_denotation(sr);
    }

    std::string as_str() const override {
        return predicate_->name_;
    }
};

class PlusRole : public Role {
  protected:
    const Role *role_;

  public:
    PlusRole(const Role *role) : Role(1 + role->complexity()), role_(role) { }
    virtual ~PlusRole() { }

    // apply Johnson's algorithm for transitive closure
    void transitive_closure(int num_objects, state_denotation_t &sd) const {
        // create adjacency lists
        std::vector<std::vector<int> > adj(num_objects);
        for( int i = 0; i < num_objects; ++i ) {
            for( int j = 0; j < num_objects; ++j ) {
                if( sd[i * num_objects + j] )
                    adj[i].emplace_back(j);
            }
        }

        // apply dfs starting from each vertex
        for( int r = 0; r < num_objects; ++r ) {
            std::vector<bool> visited(num_objects, false);
            std::vector<int> q = adj[r];
            while( !q.empty() ) {
                int i = q.back();
                q.pop_back();
                sd[r * num_objects + i] = true;
                if( !visited[i] ) {
                    visited[i] = true;
                    q.insert(q.end(), adj[i].begin(), adj[i].end());
                }
            }
        }
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            const sample_denotation_t *r = cache.find_sample_denotation(role_->as_str());
            if( r == nullptr ) return nullptr; // CHECK: BLAI HACK
            assert((r != nullptr) && (r->size() == sample.num_states()));

            sample_denotation_t nr;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const state_denotation_t *sr = (*r)[i];
                assert((sr != nullptr) && (sr->size() == sample.num_objects() * sample.num_objects()));

                state_denotation_t nsr(*sr);
                transitive_closure(sample.num_objects(), nsr);
                nr.emplace_back(cache.find_or_insert_state_denotation(nsr));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nr, as_str()) : new sample_denotation_t(nr);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: PlusRole::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }

    std::string as_str() const override {
        return std::string("Plus[") + role_->as_str() + "]";
    }
};

class StarRole : public Role {
  protected:
    const Role *role_;
    const PlusRole *plus_role_;

  public:
    StarRole(const Role *role)
      : Role(1 + role->complexity()),
        role_(role),
        plus_role_(new PlusRole(role)) {
    }
    virtual ~StarRole() {
        delete plus_role_;
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            const sample_denotation_t *pr = cache.find_sample_denotation(plus_role_->as_str());
            if( pr == nullptr ) return nullptr; // CHECK: BLAI HACK
            assert((pr != nullptr) && (pr->size() == sample.num_states()));

            sample_denotation_t nr;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const state_denotation_t *sr = (*pr)[i];
                assert((sr != nullptr) && (sr->size() == sample.num_objects() * sample.num_objects()));

                state_denotation_t nsr(*sr);
                for( int j = 0; j < sample.num_objects(); ++j ) {
                    int index = j * sample.num_objects() + j;
                    nsr[index] = true;
                }
                nr.emplace_back(cache.find_or_insert_state_denotation(nsr));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nr, as_str()) : new sample_denotation_t(nr);
        } else {
            return cached;
        }
    }
    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: StarRole::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }

    std::string as_str() const override {
        return std::string("Star[") + role_->as_str() + "]";
    }
};

class InverseRole : public Role {
  protected:
    const Role *role_;

  public:
    InverseRole(const Role *role) : Role(1 + role->complexity()), role_(role) { }
    virtual ~InverseRole() { }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample, bool use_cache) const override {
        const sample_denotation_t *cached = use_cache ? cache.find_sample_denotation(as_str()) : nullptr;
        if( cached == nullptr ) {
            const sample_denotation_t *r = cache.find_sample_denotation(role_->as_str());
            assert((r != nullptr) && (r->size() == sample.num_states()));

            sample_denotation_t nr;
            for( unsigned i = 0; i < sample.num_states(); ++i ) {
                const state_denotation_t *sr = (*r)[i];
                assert((sr != nullptr) && (sr->size() == sample.num_objects() * sample.num_objects()));

                state_denotation_t nsr(sample.num_objects() * sample.num_objects(), false);
                for( int j = 0; j < sample.num_objects(); ++j ) {
                    for( int k = 0; k < sample.num_objects(); ++k ) {
                        int index = j * sample.num_objects() + k;
                        if( (*sr)[index] ) {
                            int inv_index = k * sample.num_objects() + j;
                            nsr[inv_index] = true;
                        }
                    }
                }
                nr.emplace_back(cache.find_or_insert_state_denotation(nsr));
            }
            return use_cache ? cache.find_or_insert_sample_denotation(nr, as_str()) : new sample_denotation_t(nr);
        } else {
            return cached;
        }
    }
    virtual const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        throw std::runtime_error("Unexpected call: InverseRole::denotation(Cache&, const Sample&, const State&)");
        return nullptr;
    }

    std::string as_str() const override {
        return std::string("Inverse[") + role_->as_str() + "]";
    }
};

class Feature {
  public:
    Feature() { }
    virtual int complexity() const = 0;
    virtual int value(const Cache &cache, const Sample &sample, const State &state) const = 0;
    virtual std::string as_str() const = 0;
    friend std::ostream& operator<<(std::ostream &os, const Feature &f) {
        return os << f.as_str() << std::flush;
    }
};

class BooleanFeature : public Feature {
  protected:
    const Concept &concept_;

  public:
    BooleanFeature(const Concept &concept) : Feature(), concept_(concept) { }
    int complexity() const override {
        return concept_.complexity();
    }
    int value(const Cache &cache, const Sample &sample, const State &state) const override {
        // we look into cache for sample denotation using the concept name,
        // then index sample denotation with state id to find state denotation,
        // for finally computing cardinality (this assumes that state id is
        // index of state into sample.states())
        assert(sample.states()[state.id()].id() == state.id());
        const sample_denotation_t *d = cache.find_sample_denotation(concept_.as_str());
        assert((d != nullptr) && (d->size() == sample.num_states()));
        const state_denotation_t *sd = (*d)[state.id()];
        assert((sd != nullptr) && (sd->size() == sample.num_objects()));
        assert(sd->cardinality() < 2);
        return sd->cardinality();
    }
    std::string as_str() const override {
        return std::string("Boolean[") + concept_.as_str() + "]";
    }
};

class NumericalFeature : public Feature {
  protected:
    const Concept &concept_;

  public:
    NumericalFeature(const Concept &concept) : Feature(), concept_(concept) { }
    int complexity() const override {
        return concept_.complexity();
    }
    int value(const Cache &cache, const Sample &sample, const State &state) const override {
        // we look into cache for sample denotation using the concept name,
        // then index sample denotation with state id to find state denotation,
        // for finally computing cardinality (this assumes that state id is
        // index of state into sample.states())
        assert(sample.states()[state.id()].id() == state.id());
        const sample_denotation_t *d = cache.find_sample_denotation(concept_.as_str());
        assert((d != nullptr) && (d->size() == sample.num_states()));
        const state_denotation_t *sd = (*d)[state.id()];
        assert((sd != nullptr) && (sd->size() == sample.num_objects()));
        return sd->cardinality();
    }
    std::string as_str() const override {
        return std::string("Numerical[") + concept_.as_str() + "]";
    }
};

class Factory {
  protected:
    const std::string name_;
    std::vector<const Role*> basis_roles_;
    std::vector<const Concept*> basis_concepts_;

    int complexity_bound_;

    mutable int current_complexity_;
    mutable std::vector<const Role*> roles_;
    mutable std::vector<std::vector<const Concept*> > concepts_;

    mutable std::vector<const Feature*> features_;

  public:
    Factory(const std::string &name, int complexity_bound)
      : name_(name),
        complexity_bound_(complexity_bound),
        current_complexity_(-1) {
    }
    virtual ~Factory() { }

    const std::string& name() const {
        return name_;
    }

    void insert_basis(const Role *role) {
        basis_roles_.push_back(role);
    }
    void insert_basis(const Concept *concept) {
        basis_concepts_.push_back(concept);
    }

    int current_complexity() const {
        return current_complexity_;
    }
    void set_complexity_bound(int complexity_bound) {
        complexity_bound_ = complexity_bound;
    }

    int reset(bool remove) {
        while( roles_.size() > 0 ) {
            if( remove && (roles_.size() >= basis_roles_.size()) )
                delete roles_.back();
            roles_.pop_back();
        }

        while( concepts_.size() > 1 ) {
            if( remove ) {
                for( int k = 0; k < int(concepts_.back().size()); ++k )
                    delete concepts_.back()[k];
            }
            concepts_.pop_back();
        }
        concepts_.pop_back();

        current_complexity_ = -1;
        return current_complexity_;
    }

    int advance_step() const {
        if( current_complexity_ == -1 ) {
            assert(concepts_.empty());
            concepts_.emplace_back(basis_concepts_);
        } else {
            std::vector<const Concept*> concepts;
            for( int i = 0; i < int(concepts_.back().size()); ++i ) {
                const Concept *c = concepts_.back()[i];
                concepts.push_back(new NotConcept(c));
                for( int j = 0; j < int(roles_.size()); ++j ) {
                    const Role *r = roles_[j];
                    concepts.push_back(new ExistsConcept(c, r));
                    concepts.push_back(new ForallConcept(c, r));
                }
                for( int j = 1 + i; j < int(concepts_.back().size()); ++j ) {
                    const Concept *oc = concepts_.back()[j];
                    concepts.push_back(new AndConcept(c, oc));
                }
            }
            concepts_.emplace_back(std::move(concepts));
        }
        ++current_complexity_;
        return current_complexity_;
    }

    bool is_superfluous(Cache &cache, const sample_denotation_t *d) const {
        return cache.find_sample_denotation(*d) != nullptr;
    }
    void insert_new_denotation(Cache &cache, const std::string &name, const sample_denotation_t *d) const {
        cache.find_or_insert_sample_denotation(*d, name);
    }

    int prune_superfluous_concepts_in_last_layer(Cache &cache, const Sample &sample) const {
        if( concepts_.empty() ) return 0;

        // for each concept, check whether there is another concept with
        // same denotation in all states. For this, create a vector of
        // denotations, one for each state, and check whether there is
        // another concept with same vector. We store vectors in a hash
        // table to make the check more efficient

        int num_pruned_concepts = 0;
        for( int i = 0; i < int(concepts_.back().size()); ++i ) {
            const Concept &c = *concepts_.back()[i];
            //std::cout << "TEST-PRUNE: c=" << c.as_str() << std::endl;
            const sample_denotation_t *d = c.denotation(cache, sample, false);
            //std::cout << "hola0" << std::endl;
            if( !is_superfluous(cache, d) ) {
                //std::cout << "hola1: " << d->size() << std::endl;
                insert_new_denotation(cache, c.as_str(), d);
                //std::cout << "hola2" << std::endl;
            } else {
                //std::cout << "hola3" << std::endl;
                // make sure we do not eliminate a basis
                if( concepts_.size() > 1 ) delete concepts_.back()[i];
                concepts_.back()[i] = concepts_.back().back();
                concepts_.back().pop_back();
                --i;
                ++num_pruned_concepts;
                //std::cout << "hola7" << std::endl;
            }
            delete d;
        }
        return num_pruned_concepts;
    }

    void generate_basis(const Sample &sample) {
        insert_basis(new UniversalConcept);
        for( int i = 0; i < int(sample.num_predicates()); ++i ) {
            const Predicate *predicate = &sample.predicates()[i];
            if( predicate->arity() == 1 ) {
                insert_basis(new PrimitiveConcept(predicate));
            } else if( predicate->arity() == 2 ) {
                insert_basis(new PrimitiveRole(predicate));
            }
        }
        std::cout << "BASIS: #concepts=" << basis_concepts_.size()
                  << ", #roles=" << basis_roles_.size()
                  << std::endl;
    }

    int generate_roles(Cache &cache, const Sample *sample = nullptr) const {
        if( roles_.empty() ) {
            roles_.insert(roles_.end(), basis_roles_.begin(), basis_roles_.end());
            for( int i = 0; i < int(basis_roles_.size()); ++i ) {
                const Role *role = basis_roles_[i];
                roles_.push_back(new PlusRole(role));
                roles_.push_back(new StarRole(role));
                roles_.push_back(new InverseRole(role));
                const Role *irole = roles_.back();
                roles_.push_back(new PlusRole(irole));
                roles_.push_back(new StarRole(irole));
            }
        }

        // create denotations for roles
        int num_pruned_roles = 0;
        if( sample != nullptr ) {
            for( int i = 0; i < int(roles_.size()); ++i ) {
                const Role &role = *roles_[i];
                const sample_denotation_t *r = role.denotation(cache, *sample, false);
                if( (r != nullptr) && !is_superfluous(cache, r) ) { // CHECK: BLAI HACK
                    insert_new_denotation(cache, role.as_str(), r);
                } else {
                    // make sure we do not eliminate a basis
                    if( i >= basis_roles_.size() ) delete roles_[i];
                    for( int j = i; 1 + j < int(roles_.size()); ++j )
                        roles_[j] = roles_[1 + j];
                    roles_.pop_back();
                    //roles_[i] = roles_.back();
                    //roles_.pop_back();
                    --i;
                    ++num_pruned_roles;
                }
                delete r;
            }
        }

        std::cout << "ROLES: #roles=" << roles_.size() << ", #pruned-roles=" << num_pruned_roles << std::endl;
        return roles_.size();
    }

    int generate_concepts(Cache &cache, const Sample *sample = nullptr, bool prune = false) const {
        std::cout << "DL::Factory: name="
                  << name_
                  << ", generate_concepts()"
                  << ", current-complexity=" << current_complexity_
                  << ", complexity-bound=" << complexity_bound_
                  << std::endl;

        int num_concepts = 0;
        while( current_complexity_ < complexity_bound_ ) {
            std::cout << "DL::Factory: iteration:"
                      << " current-complexity="
                      << current_complexity_
                      << ", #concepts-in-layer="
                      << (concepts_.empty() ? 0 : concepts_.back().size())
                      << std::endl;

            advance_step();
            std::cout << "DL::Factory: advance-step:"
                      << " #concepts-in-layer="
                      << concepts_.back().size()
                      << std::flush;
            if( prune && (sample != nullptr) ) {
                int number_pruned_concepts = prune_superfluous_concepts_in_last_layer(cache, *sample);
                std::cout << ", #pruned-concepts-in-layer="
                          << number_pruned_concepts
                          << std::flush;
            }
            std::cout << std::endl;
            num_concepts += concepts_.empty() ? 0 : concepts_.back().size();
        }
        std::cout << "DL::Factory: name=" << name_ << ", #concepts=" << num_concepts << std::endl;
        return current_complexity_;
    }

    int generate_features(const Cache &cache, const Sample &sample) const {
        // create boolean/numerical features from concepts
        int num_boolean_features = 0;
        for( int layer = 0; layer < int(concepts_.size()); ++layer ) {
            for( int i = 0; i < int(concepts_[layer].size()); ++i ) {
                const Concept &c = *concepts_[layer][i];
                const sample_denotation_t *d = cache.find_sample_denotation(c.as_str());
                assert((d != nullptr) && (d->size() == sample.num_states()));
                bool boolean_feature = true;
                for( int j = 0; boolean_feature && (j < sample.num_states()); ++j ) {
                    assert(((*d)[j] != nullptr) && ((*d)[j]->size() == sample.num_objects()));
                    int cardinality = (*d)[j]->cardinality();
                    boolean_feature = cardinality < 2;
                }
                num_boolean_features += boolean_feature;
                features_.emplace_back(boolean_feature ? static_cast<Feature*>(new BooleanFeature(c)) : static_cast<Feature*>(new NumericalFeature(c)));
            }
        }

        // create distance features
        int num_distance_features = 0;
        //throw std::runtime_error("TODO: UNIMPLEMENTED: Factory::generate_features()");

        std::cout << "FEATURES: #features=" << features_.size()
                  << ", #boolean-features=" << num_boolean_features
                  << ", #numerical-features=" << features_.size() - num_boolean_features - num_distance_features
                  << ", #distance-features=" << num_distance_features
                  << std::endl;

        return features_.size();
    }

    void output_feature_matrix(std::ostream &os, const Cache &cache, const Sample &sample) const {
        for( unsigned i = 0; i < sample.num_states(); ++i ) {
            const State &state = sample.state(i);
            std::vector<std::pair<int, int> > non_zero_values;
            for( int j = 0; j < int(features_.size()); ++j ) {
                const Feature &f = *features_[j];
                int value = f.value(cache, sample, state);
                if( value > 0 )
                    non_zero_values.emplace_back(j, value);
            }
            if( !non_zero_values.empty() ) {
                os << i << " " << non_zero_values.size();
                for( int j = 0; j < int(non_zero_values.size()); ++j ) {
                    os << " " << non_zero_values[j].first
                       << " " << non_zero_values[j].second;
                }
                os << std::endl;
            }
        }
    }
};

}; // DL namespace

}; // SLTP namespace

#endif


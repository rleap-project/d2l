
#pragma once


#include <cassert>
#include <iostream>

#include <common/base.h>
#include "cache.hxx"


namespace sltp::dl {

const unsigned PRIMITIVE_COMPLEXITY = 1;

//! A common base class for concepts and roles
class DLBaseElement {
private:
    static unsigned long global_id;

protected:
    const unsigned long id_;
    unsigned complexity_;

public:
    explicit DLBaseElement(int complexity) : id_(global_id++), complexity_(complexity) { }

    unsigned complexity() const { return complexity_; }

    unsigned long id() const { return id_; }

    //! By default we raise an exception here, as we want users to use the method that returns the denotation
    //! over the whole sample, which will be more efficient. Some subclasses though will override this, mostly
    //! Primitive concepts and roles.
    virtual const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const {
        throw std::runtime_error("Unexpected call to DLBaseElement::denotation(Cache&, const Sample&, const State&)");
    }

    //! Compute the full sample denotation for the current DL element
    virtual const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const = 0;

    //! Return a string representation of the concept or role
    virtual std::string str() const = 0;

    //! Return a string representation of the concept or role that includes its complexity.
    std::string fullstr() const {
        return str() + "[id=" + std::to_string(id()) + ",k=" + std::to_string(complexity_) + "]";
    }

    void force_complexity(int c) {
        complexity_ = c;
    }

    virtual const DLBaseElement* clone() const = 0;

    friend std::ostream& operator<<(std::ostream &os, const DLBaseElement &base) {
        return os << base.fullstr() << std::flush;
    }
};

class Concept : public DLBaseElement {
public:
    explicit Concept(int complexity) : DLBaseElement(complexity) { }
    virtual ~Concept() = default;
    const Concept* clone() const override = 0;
};

class Role : public DLBaseElement {
public:
    explicit Role(int complexity) : DLBaseElement(complexity) { }
    virtual ~Role() = default;
    const Role* clone() const override = 0;
};

class PrimitiveConcept : public Concept {
protected:
    const Predicate *predicate_;

public:
    explicit PrimitiveConcept(const Predicate *predicate) : Concept(PRIMITIVE_COMPLEXITY), predicate_(predicate) { }
    ~PrimitiveConcept() override = default;
    const Concept* clone() const override {
        return new PrimitiveConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            res->emplace_back(denotation(cache, sample, sample.state(i)));
        }
        return res;
    }

    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        state_denotation_t sd(sample.num_objects(state.id()), false);
        for( int i = 0; i < int(state.atoms().size()); ++i ) {
            atom_id_t id = state.atoms()[i];
            const Atom &atom = sample.atom(state.id(), id);
            if( atom.is_instance(*predicate_) ) {
                assert(atom.objects().size() == 1);
                object_id_t index = atom.object(0);
                assert(index < sample.num_objects(state.id()));
                sd[index] = true;
            }
        }
        return cache.find_or_insert_state_denotation(sd);
    }
    std::string str() const override {
        return predicate_->name_;
    }
};


class NominalConcept : public Concept {
protected:
    const std::string& name_;

public:
    explicit NominalConcept(const std::string &name) : Concept(1), name_(name) {}
    ~NominalConcept() override = default;

    const Concept* clone() const override {
        return new NominalConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            res->emplace_back(denotation(cache, sample, sample.state(i)));
        }
        return res;
    }


    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        const auto& oidx = sample.instance(state.id()).object_index();
        object_id_t id = oidx.left.at(name_);
        assert(id < sample.num_objects(state.id()));

        state_denotation_t sd(sample.num_objects(state.id()), false);
        sd[id] = true;

        return cache.find_or_insert_state_denotation(sd);
    }

    std::string str() const override {
        return std::string("Nominal(") + name_ + ")";
    }
};

class UniversalConcept : public Concept {
public:
    UniversalConcept() : Concept(0) {}

    ~UniversalConcept() override = default;

    const Concept* clone() const override {
        return new UniversalConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            const auto n = sample.num_objects(i);

            state_denotation_t sd(n, true);
            res->emplace_back(cache.find_or_insert_state_denotation(sd));
        }
        return res;
    }

    std::string str() const override {
        return "<universe>";
    }
};


class EmptyConcept : public Concept {
public:
    EmptyConcept() : Concept(0) { }
    ~EmptyConcept() override = default;
    const Concept* clone() const override {
        return new EmptyConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            state_denotation_t std(sample.num_objects(i), false);
            res->emplace_back(cache.find_or_insert_state_denotation(std));
        }
        return res;
    }

    std::string str() const override {
        return "<empty>";
    }
};

class AndConcept : public Concept {
protected:
    const Concept *concept1_;
    const Concept *concept2_;

public:
    AndConcept(const Concept *concept1, const Concept *concept2) :
            Concept(1 + concept1->complexity() + concept2->complexity()),
            concept1_(concept1),
            concept2_(concept2) {
    }
    ~AndConcept() override = default;
    const Concept* clone() const override {
        return new AndConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*concept1_, m);
        const sample_denotation_t& sd_sub2 = cache.find_sample_denotation(*concept2_, m);

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for( int i = 0; i < m; ++i ) {
            const auto n = sample.num_objects(i);
            const auto& sd1 = sd_sub1.get(i, n);
            const auto& sd2 = sd_sub2.get(i, n);

            state_denotation_t nsd(n, false);
            for (int j = 0; j < n; ++j) nsd[j] = sd1[j] && sd2[j];
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    std::string str() const override {
        return std::string("And(") + concept1_->str() + "," + concept2_->str() + ")";
    }
};


class OrConcept : public Concept {
protected:
    const Concept *concept1_;
    const Concept *concept2_;

public:
    OrConcept(const Concept *concept1, const Concept *concept2) :
            Concept(1 + concept1->complexity() + concept2->complexity()),
            concept1_(concept1),
            concept2_(concept2) {
    }
    ~OrConcept() override = default;
    const Concept* clone() const override {
        return new OrConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*concept1_, m);
        const sample_denotation_t& sd_sub2 = cache.find_sample_denotation(*concept2_, m);

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for( int i = 0; i < m; ++i ) {
            const auto n = sample.num_objects(i);
            const auto& sd1 = sd_sub1.get(i, n);
            const auto& sd2 = sd_sub2.get(i, n);

            state_denotation_t nsd(n, false);
            for (int j = 0; j < n; ++j) nsd[j] = sd1[j] || sd2[j];
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    std::string str() const override {
        return std::string("Or(") + concept1_->str() + "," + concept2_->str() + ")";
    }
};

class NotConcept : public Concept {
protected:
    const Concept *concept_;

public:
    explicit NotConcept(const Concept *concept)
        : Concept(1 + concept->complexity()), concept_(concept)
    {}
    ~NotConcept() override = default;
    const Concept* clone() const override {
        return new NotConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*concept_, m);

        auto *res = new sample_denotation_t();
        res->reserve(m);
        for (unsigned i = 0; i < m; ++i) {
            const auto n = sample.num_objects(i);
            const auto& std_sub1 = sd_sub1.get(i, n);

            state_denotation_t nsd(n, false);
            for (unsigned j = 0; j < n; ++j) nsd[j] = !std_sub1[j];
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    std::string str() const override {
        return std::string("Not(") + concept_->str() + ")";
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
    ~ExistsConcept() override = default;
    const Concept* clone() const override {
        return new ExistsConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*concept_, m);
        const sample_denotation_t& sd_sub2 = cache.find_sample_denotation(*role_, m);

        auto *res = new sample_denotation_t();
        res->reserve(m);
        for( unsigned i = 0; i < m; ++i ) {
            const auto n = sample.num_objects(i);
            const auto& c_den = sd_sub1.get(i, n);
            const auto& r_den = sd_sub2.get(i, n*n);

            state_denotation_t nsd(n, false);
            for (unsigned x = 0; x < n; ++x) {
                // x makes it into the denotation if there is an y such that y in c_den and (x,y) in r_den
                for (unsigned y = 0; y < n; ++y) {
                    if(c_den[y]) {
                        auto x_y = x * n + y;
                        if (r_den[x_y]) {
                            nsd[x] = true;
                            break;
                        }
                    }
                }
            }
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    std::string str() const override {
        return std::string("Exists(") + role_->str() + "," + concept_->str() + ")";
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
    ~ForallConcept() override = default;
    const Concept* clone() const override {
        return new ForallConcept(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*concept_, m);
        const sample_denotation_t& sd_sub2 = cache.find_sample_denotation(*role_, m);

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for( int i = 0; i < m; ++i ) {
            const auto n = sample.num_objects(i);
            const auto& c_den = sd_sub1.get(i, n);
            const auto& r_den = sd_sub2.get(i, n*n);

            state_denotation_t nsd(n, true);
            for (unsigned x = 0; x < n; ++x) {
                // x does *not* make it into the denotation if there is an y
                // such that y not in c_den and (x,y) in r_den
                for (unsigned y = 0; y < n; ++y) {
                    if (!c_den[y]) {
                        auto x_y = x * n + y;
                        if (r_den[x_y]) {
                            nsd[x] = false;
                            break;
                        }
                    }
                }
            }
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    std::string str() const override {
        return std::string("Forall(") + role_->str() + "," + concept_->str() + ")";
    }
};


class EqualConcept : public Concept {
protected:
    const Role *r1_;
    const Role *r2_;

public:
    EqualConcept(const Role *r1, const Role *r2)
            : Concept(1 + r1->complexity() + r2->complexity()),
              r1_(r1),
              r2_(r2) {
    }
    ~EqualConcept() override = default;
    const Concept* clone() const override { return new EqualConcept(*this); }


    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*r1_, m);
        const sample_denotation_t& sd_sub2 = cache.find_sample_denotation(*r2_, m);

        auto *res = new sample_denotation_t();
        res->reserve(m);
        for( int i = 0; i < m; ++i ) {
            const auto n = sample.num_objects(i);
            const auto& sd1 = sd_sub1.get(i, n*n);
            const auto& sd2 = sd_sub2.get(i, n*n);

            state_denotation_t nsd(n, false);
            for (int x = 0; x < n; ++x) {
                // If the set of y such that (x, y) in sd1 is equal to the set of z such that (x, z) in sd2,
                // then x makes it into the denotation of this concept
                bool in_denotation = true;
                for (int z = 0; z < n; ++z) {
                    auto idx = x * n + z;
                    if (sd1[idx] != sd2[idx]) {
                        in_denotation = false;
                        break;
                    }
                }

                if (in_denotation) {
                    nsd[x] = true;
                }
            }
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    std::string str() const override {
        return std::string("Equal(") + r1_->str() + "," + r2_->str() + ")";
    }
};


class PrimitiveRole : public Role {
protected:
    const Predicate *predicate_;

public:
    explicit PrimitiveRole(const Predicate *predicate) : Role(PRIMITIVE_COMPLEXITY), predicate_(predicate) { }
    ~PrimitiveRole() override = default;
    const Role* clone() const override {
        return new PrimitiveRole(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for( int i = 0; i < m; ++i ) {
            res->emplace_back(denotation(cache, sample, sample.state(i)));
        }
        return res;
    }

    const state_denotation_t* denotation(Cache &cache, const Sample &sample, const State &state) const override {
        const auto n = sample.num_objects(state.id());
        state_denotation_t sr(n * n, false);
        for (const auto atom_id : state.atoms()) {
            const Atom &atom = sample.atom(state.id(), atom_id);
            if( atom.is_instance(*predicate_) ) {
                assert(atom.objects().size() == 2);
                unsigned index = atom.object(0) * n + atom.object(1);
                assert(index < n * n);
                sr[index] = true;
            }
        }
        return cache.find_or_insert_state_denotation(sr);
    }

    const Predicate* predicate() const { return predicate_; }

    std::string str() const override {
        return predicate_->name_;
    }
};

class PlusRole : public Role {
protected:
    const Role *role_;

public:
    explicit PlusRole(const Role *role) : Role(1 + role->complexity()), role_(role) { }
    ~PlusRole() override = default;
    const Role* clone() const override {
        return new PlusRole(*this);
    }

    // apply Johnson's algorithm for transitive closure
    static void transitive_closure(int num_objects, state_denotation_t &sd) {
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

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*role_, m);

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            const auto n = sample.num_objects(i);
            const auto& sd1 = sd_sub1.get(i, n*n);

            state_denotation_t nsd(sd1);
            transitive_closure(n, nsd);
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    const Role* role() const { return role_; }

    std::string str() const override {
        // ATM let us call these Star(X) to get the same output than the Python module
        return std::string("Star(") + role_->str() + ")";
    }
};

class StarRole : public Role {
protected:
    const Role *role_;
    const PlusRole *plus_role_;

public:
    explicit StarRole(const Role *role)
            : Role(1 + role->complexity()),
              role_(role),
              plus_role_(new PlusRole(role)) {
    }
    ~StarRole() override {
        delete plus_role_;
    }

    const Role* clone() const override {
        return new StarRole(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        throw std::runtime_error("UNIMPLEMENTED");
    }

    const Role* role() const { return role_; }

    std::string str() const override {
        return std::string("Star(") + role_->str() + ")";
    }
};

class InverseRole : public Role {
protected:
    const Role *role_;

public:
    explicit InverseRole(const Role *role) : Role(1 + role->complexity()), role_(role) { }

    ~InverseRole() override = default;

    const Role* clone() const override {
        return new InverseRole(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*role_, m);

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            const auto n = sample.num_objects(i);
            const auto& sr = sd_sub1.get(i, n*n);

            state_denotation_t nsd(n*n, false);
            for (unsigned j = 0; j < n; ++j) {
                for (unsigned k = 0; k < n; ++k) {
                    unsigned index = j * n + k;
                    if (sr[index]) {
                        unsigned inv_index = k * n + j;
                        nsd[inv_index] = true;
                    }
                }
            }
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }


    const Role* role() const { return role_; }

    std::string str() const override {
        return std::string("Inverse(") + role_->str() + ")";
    }
};

// RoleRestriction are only used for distance features
// and thus they are generated when generating such features
class RoleRestriction : public Role {
protected:
    const Role *role_;
    const Concept *restriction_;

public:
    RoleRestriction(const Role *role, const Concept *restriction)
            : Role(1 + role->complexity() + restriction->complexity()),
              role_(role),
              restriction_(restriction) {
    }

    ~RoleRestriction() override = default;

    const Role* clone() const override {
        return new RoleRestriction(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*role_, m);
        const sample_denotation_t& sd_sub2 = cache.find_sample_denotation(*restriction_, m);

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            const auto n = sample.num_objects(i);
            const auto& sr = sd_sub1.get(i, n*n);
            const auto& sd = sd_sub2.get(i, n);

            state_denotation_t nsd(sr);
            for (int j = 0; j < n*n; ++j) {
                if( nsd[j] ) {
                    //int src = j / n;
                    int dst = j % n;
                    nsd[j] = sd[dst];
                }
            }
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    const Role* role() const { return role_; }

    std::string str() const override {
        return std::string("Restrict(") + role_->str() + "," + restriction_->str() + ")";
    }
};

//! R - R' represents the set of pairs (a,b) such that R(a,b) and not R'(a,b)
//! Makes sense e.g. when R is a goal predicate
class RoleDifference : public Role {
protected:
    const Role *r1_;
    const Role *r2_;

public:
    RoleDifference(const Role *r1, const Role *r2)
            : Role(1 + r1->complexity() + r2->complexity()),
              r1_(r1),
              r2_(r2) {
    }
    ~RoleDifference() override = default;
    const Role* clone() const override {
        return new RoleDifference(*this);
    }

    const sample_denotation_t* denotation(Cache &cache, const Sample &sample) const override {
        const auto m = sample.num_states();
        const sample_denotation_t& sd_sub1 = cache.find_sample_denotation(*r1_, m);
        const sample_denotation_t& sd_sub2 = cache.find_sample_denotation(*r2_, m);

        auto* res = new sample_denotation_t();
        res->reserve(m);
        for (int i = 0; i < m; ++i) {
            const auto n = sample.num_objects(i);
            const auto& sd1 = sd_sub1.get(i, n*n);
            const auto& sd2 = sd_sub2.get(i, n*n);

            state_denotation_t nsd(n*n, false);
            for (int x = 0; x < n*n; ++x) {
                if (sd1[x] && !sd2[x]) {
                    nsd[x] = true;
                }
            }
            res->emplace_back(cache.find_or_insert_state_denotation(nsd));
        }
        return res;
    }

    std::string str() const override {
        return std::string("RoleDifference(") + r1_->str() + "," + r2_->str() + ")";
    }
};

class Feature {
public:
    using value_t = int;
    static const value_t infty = std::numeric_limits<int>::max();
    
    Feature() = default;
    virtual ~Feature() = default;
    virtual const Feature* clone() const = 0;

    virtual unsigned complexity() const = 0;
    virtual value_t value(const Cache &cache, const Sample &sample, const State &state) const = 0;
    virtual std::string as_str() const = 0;

    std::string as_str_with_complexity() const {
        return std::to_string(complexity()) + "." + as_str();
    }

    friend std::ostream& operator<<(std::ostream &os, const Feature &f) {
        return os << f.as_str() << std::flush;
    }

    virtual bool is_boolean() const = 0;
};

class NullaryAtomFeature : public Feature {
protected:
    const Predicate* predicate_;

public:
    explicit NullaryAtomFeature(const Predicate* predicate) : predicate_(predicate) { }
    ~NullaryAtomFeature() override = default;

    const Feature* clone() const override {
        return new NullaryAtomFeature(*this);
    }

    unsigned complexity() const override { // Nullary atoms have complexity 0 by definition
        return 1;
    }

    value_t value(const Cache &cache, const Sample &sample, const State &state) const override {
        // Return true (1) iff some atom in the state has the coindiding predicate ID with the predicate of the
        // nullary atom, since there can only be one single atom with this predicate
        // TODO: This is unnecessarily expensive, and it is not cached anywhere
        for (const auto& atom_id:state.atoms()) {
            const Atom &atom = sample.atom(state.id(), atom_id);
            if( atom.is_instance(*predicate_)) {
                return 1;
            }
        }
        return 0;
    }

    std::string as_str() const override {
        return std::string("Atom[") + predicate_->name_ + "]";
    }

    bool is_boolean() const override { return true; }
};

class BooleanFeature : public Feature {
protected:
    const Concept *concept_;

public:
    explicit BooleanFeature(const Concept *concept) : Feature(), concept_(concept) { }

    ~BooleanFeature() override = default;

    const Feature* clone() const override {
        return new BooleanFeature(concept_);
    }

    unsigned complexity() const override {
        return concept_->complexity();
    }

    value_t value(const Cache &cache, const Sample &sample, const State &state) const override {
        // we retrieve the sample denotation from the cache, then the state denotation from the sample denotation,
        // and compute the cardinality (this assumes that state id is index of state into sample.states())
        const auto m = sample.num_states();
        const sample_denotation_t& d = cache.find_sample_denotation(*concept_, m);
        const state_denotation_t& std = d.get(state.id(), sample.num_objects(state.id()));
        assert(std.cardinality() < 2);
        return std.cardinality();
    }

    std::string as_str() const override {
        return std::string("Bool[") + concept_->str() + "]";
    }

    bool is_boolean() const override { return true; }
};

class NumericalFeature : public Feature {
protected:
    const Concept *concept_;

public:
    explicit NumericalFeature(const Concept *concept) : Feature(), concept_(concept) { }

    ~NumericalFeature() override = default;

    const Feature* clone() const override {
        return new NumericalFeature(concept_);
    }

    unsigned complexity() const override {
        return concept_->complexity();
    }

    value_t value(const Cache &cache, const Sample &sample, const State &state) const override {
        // we retrieve the sample denotation from the cache, then the state denotation from the sample denotation,
        // and compute the cardinality (this assumes that state id is index of state into sample.states())
        const auto m = sample.num_states();
        const sample_denotation_t& d = cache.find_sample_denotation(*concept_, m);
        const state_denotation_t& std = d.get(state.id(), sample.num_objects(state.id()));
        return std.cardinality();
    }

    std::string as_str() const override {
        return std::string("Num[") + concept_->str() + "]";
    }

    bool is_boolean() const override { return false; }
};

class DistanceFeature : public Feature {
protected:
    const Concept *start_;
    const Concept *end_;
    const Role *role_;

    mutable bool valid_cache_;
    mutable std::vector<int> cached_distances_;

public:
    DistanceFeature(const Concept *start, const Concept *end, const Role *role)
            : Feature(),
              start_(start),
              end_(end),
              role_(role),
              valid_cache_(false)
    {}
    ~DistanceFeature() override = default;
    const Feature* clone() const override {
        auto *f = new DistanceFeature(start_, end_, role_);
        f->valid_cache_ = valid_cache_;
        f->cached_distances_ = cached_distances_;
        return f;
    }

    unsigned complexity() const override {
        return 1 + start_->complexity() + end_->complexity() + role_->complexity();
    }
    value_t value(const Cache &cache, const Sample &sample, const State &state) const override;

    std::string as_str() const override {
        return std::string("Dist[") + start_->str() + ";" + role_->str() + ";" + end_->str() + "]";
    }

    bool is_boolean() const override { return false; }
};


class ConditionalFeature : public Feature {
protected:
    const Feature* condition_;
    const Feature* body_;

public:
    ConditionalFeature(const Feature* condition, const Feature* body) :
            Feature(), condition_(condition), body_(body) { }

    ~ConditionalFeature() override = default;
    const Feature* clone() const override {
        return new ConditionalFeature(condition_, body_);
    }

    unsigned complexity() const override {
        return 1 + condition_->complexity() + body_->complexity();
    }

    value_t value(const Cache &cache, const Sample &sample, const State &state) const override {
        return value_from_components(cache, sample, state, condition_, body_);
    }

    static int value_from_components(const Cache &cache, const Sample &sample, const State &state,
                                     const Feature* condition, const Feature* body) {
        const auto condval = condition->value(cache, sample, state);
        return (condval) ? body->value(cache, sample, state) : std::numeric_limits<int>::max();
    }

    std::string as_str() const override {
        return std::string("If{") + condition_->as_str() + "}{" + body_->as_str() + "}{Infty}";
    }

    bool is_boolean() const override { return false; }
};


//! A feature with value f1 < f2
class DifferenceFeature : public Feature {
protected:
    const Feature* f1;
    const Feature* f2;

public:
    DifferenceFeature(const Feature* f1, const Feature* f2) :
            Feature(), f1(f1), f2(f2)
    {}

    ~DifferenceFeature() override = default;
    const Feature* clone() const override {
        return new DifferenceFeature(f1, f2);
    }

    unsigned complexity() const override {
        return 1 + f1->complexity() + f2->complexity();
    }

    value_t value(const Cache &cache, const Sample &sample, const State &state) const override {
        const auto f1val = f1->value(cache, sample, state);
        const auto f2val = f2->value(cache, sample, state);
        return f1val < f2val;
    }

    std::string as_str() const override {
        return std::string("LessThan{") + f1->as_str() + "}{" + f2->as_str() + "}";
    }

    bool is_boolean() const override { return true; }
};



} // namespaces

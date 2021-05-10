
#pragma once

#include <string>
#include <iostream>
#include <vector>

#include <boost/bimap.hpp>

#include <common/utils.h>


namespace sltp {

class Sample;

using object_id_t = unsigned;
using predicate_id_t = unsigned;
using atom_id_t = unsigned;
using state_id_t = uint32_t;
using state_pair = std::pair<state_id_t, state_id_t>;
using transition_id_t = uint32_t;


// We represent states as subets of atoms
// An atom is a predicate and a vector of objects to ground the predicates.
class Object {
protected:
    const object_id_t id_;
    const std::string name_;

public:
    Object(unsigned id, std::string name) : id_(id), name_(std::move(name)) { }
    [[nodiscard]] int id() const {
        return id_;
    }
    [[nodiscard]] const std::string& as_str() const {
        return name_;
    }
    friend std::ostream& operator<<(std::ostream &os, const Object &obj) {
        return os << obj.as_str() << std::flush;
    }
};

struct Predicate {
    const predicate_id_t id_;
    const std::string name_;
    const int arity_;
    Predicate(unsigned id, std::string name, int arity)
            : id_(id), name_(std::move(name)), arity_(arity) {
    }
    [[nodiscard]] predicate_id_t id() const {
        return id_;
    }
    [[nodiscard]] const std::string& name() const {
        return name_;
    }
    [[nodiscard]] int arity() const {
        return arity_;
    }
    std::string as_str(const std::vector<object_id_t> *objects) const {
        std::string str = name_ + "(";
        if( objects == nullptr ) {
            for( int i = 0; i < arity_; ++i ) {
                str += std::string("x") + std::to_string(1 + i);
                if( 1 + i < arity_ ) str += ",";
            }
        } else {
            assert(objects->size() == arity_);
            for( int i = 0; i < arity_; ++i ) {
                //str += (*objects)[i]->str();
                str += std::to_string((*objects)[i]);
                if( 1 + i < arity_ ) str += ",";
            }
        }
        return str + ")";
    }
    friend std::ostream& operator<<(std::ostream &os, const Predicate &pred) {
        return os << pred.as_str(nullptr) << std::flush;
    }
};

class Atom {
protected:
    const predicate_id_t predicate_;
    const std::vector<object_id_t> objects_;

public:
    Atom(const predicate_id_t &predicate, std::vector<object_id_t> &&objects)
            : predicate_(predicate), objects_(std::move(objects)) {
    }

    [[nodiscard]] predicate_id_t pred_id() const {
        return predicate_;
    }
    [[nodiscard]] const std::vector<object_id_t>& objects() const {
        return objects_;
    }

    // Return the i-th object of the current atom
    [[nodiscard]] object_id_t object(int i) const {
        return objects_.at(i);
    }

    [[nodiscard]] bool is_instance(const Predicate &predicate) const {
        return predicate_ == predicate.id();
    }

    [[nodiscard]] std::vector<unsigned> data() const {
        std::vector<unsigned> res(1, predicate_);
        res.insert(res.end(), objects_.begin(), objects_.end());
        return res;
    }

    [[nodiscard]] std::string as_str(const Sample &sample) const;
};

// An instance stores information shared by the states that
// belong to the instance: objects and atoms mostly
class Instance {
public:
    // map from object name to object id in instance
//    using ObjectIndex = std::unordered_map<std::string, object_id_t>;
    using ObjectIndex = boost::bimap<std::string, object_id_t>;
    // map from atom of the form <pred_id, oid_1, ..., oid_n> to atom id in instance
    using AtomIndex = std::unordered_map<std::vector<unsigned>, atom_id_t, utils::container_hash<std::vector<unsigned> > >;

    const unsigned id;

protected:
    const std::vector<Object> objects_;
    const std::vector<Atom> atoms_;

    // mapping from object names to their ID in the sample
    ObjectIndex object_index_;

    // mapping from <predicate name, obj_name, ..., obj_name> to the ID of the corresponding GroundPredicate
    AtomIndex atom_index_;

public:
    Instance(unsigned id,
             std::vector<Object> &&objects,
             std::vector<Atom> &&atoms,
             ObjectIndex &&object_index,
             AtomIndex &&atom_index)
            :
            id(id),
            objects_(std::move(objects)),
            atoms_(std::move(atoms)),
            object_index_(std::move(object_index)),
            atom_index_(std::move(atom_index))
    {
    }

    Instance(const Instance& ins) = default;
    Instance(Instance &&ins) = default;
    ~Instance() = default;

    unsigned num_objects() const {
        return (unsigned) objects_.size();
    }
    const Atom& atom(unsigned id_) const {
        return atoms_.at(id_);
    }

    const ObjectIndex& object_index() const {
        return object_index_;
    }
    const AtomIndex& atom_index() const {
        return atom_index_;
    }
};

// A state is a collections of atoms
class State {
protected:
    const unsigned instance_id_;
    const state_id_t id_;
    std::vector<atom_id_t> atoms_;

public:
    explicit State(const unsigned instance_id, unsigned id, std::vector<atom_id_t> &&atoms)
            : instance_id_(instance_id), id_(id), atoms_(std::move(atoms)) {
    }
    State(const State &state) = default;
    State(State &&state) = default;

    [[nodiscard]] unsigned id() const { return id_; }

    [[nodiscard]] unsigned instance_id() const { return instance_id_; }

    [[nodiscard]] const std::vector<atom_id_t>& atoms() const {
        return atoms_;
    }
};

//! A sample is a bunch of states and transitions among them. The sample contains the predicates used in the states,
//! the objects, and the atoms
class Sample {
public:
    using PredicateIndex = std::unordered_map<std::string, predicate_id_t>;

protected:
    const std::vector<Predicate> predicates_;
    const std::vector<Instance> instances_;
    const std::vector<State> states_;

    // The IDs of predicates that are mentioned in the goal
    const std::vector<predicate_id_t> goal_predicates_;

    // mapping from predicate names to their ID in the sample
    PredicateIndex predicate_index_;

    Sample(std::vector<Predicate> &&predicates,
           std::vector<Instance> &&instances,
           std::vector<State> &&states,
           std::vector<predicate_id_t> &&goal_predicates,
           PredicateIndex &&predicate_index)
            : predicates_(std::move(predicates)),
              instances_(std::move(instances)),
              states_(std::move(states)),
              goal_predicates_(std::move(goal_predicates)),
              predicate_index_(std::move(predicate_index))
   {}

public:
    Sample(const Sample &sample) = default;
    Sample(Sample &&sample) = default;
    ~Sample() = default;

    const Instance& instance(unsigned sid) const {
        return instances_.at(instance_id(sid));
    }

    unsigned instance_id(unsigned sid) const {
        return state(sid).instance_id();
    }

    unsigned num_objects(unsigned sid) const {
        return instance(sid).num_objects();
    }

    const Atom& atom(unsigned sid, atom_id_t id) const {
        return instance(sid).atom(id);
    }

    std::size_t num_instances() const {
        return instances_.size();
    }
    std::size_t num_predicates() const {
        return predicates_.size();
    }
    std::size_t num_states() const {
        return states_.size();
    }

    const std::vector<Predicate>& predicates() const {
        return predicates_;
    }
    const std::vector<predicate_id_t>& goal_predicates() const {
        return goal_predicates_;
    }
    const std::vector<State>& states() const {
        return states_;
    }

    const Predicate& predicate(predicate_id_t id) const {
        return predicates_.at(id);
    }
    const State& state(unsigned sid) const {
        return states_.at(sid);
    }
    const PredicateIndex& predicate_index() const {
        return predicate_index_;
    }

    // factory method - reads sample from serialized data
    static Sample read(std::istream &is);
};

}


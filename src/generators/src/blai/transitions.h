
#pragma once

#include <cassert>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <common/base.h>


namespace sltp {

class TransitionSample {
protected:
    const std::size_t num_states_;
    const std::size_t num_transitions_;

    //! A mapping from pairs of state to the assigned transition id
    std::unordered_map<state_pair, unsigned, boost::hash<state_pair>> transition_ids_;

    //! The reverse mapping: a vector from transition id to corresponding pair of states
    std::vector<state_pair> transition_ids_inv_;

    //! trdata_[s] contains the IDs of all neighbors of s in the state space
    std::vector<std::vector<unsigned>> trdata_;

    std::vector<bool> is_state_expanded_;
    std::vector<bool> is_state_alive_;
    std::vector<bool> is_state_goal_;
    std::vector<bool> is_state_unsolvable_;
    std::vector<int> hstar_;

    std::vector<unsigned> alive_states_;
    std::vector<unsigned> goal_states_;
    std::vector<unsigned> unsolvable_states_;

    using transition_set_t = std::unordered_set<unsigned>;
    transition_set_t positive_transitions_;
    transition_set_t negative_transitions_;

public:
    TransitionSample(std::size_t num_states, std::size_t num_transitions)
            : num_states_(num_states),
              num_transitions_(num_transitions),
              trdata_(num_states),
              is_state_expanded_(num_states, false),
              is_state_alive_(num_states, false),
              is_state_goal_(num_states, false),
              is_state_unsolvable_(num_states, false),
              hstar_(num_states, -2),
              alive_states_(),
              goal_states_(),
              unsolvable_states_()
    {
        if (num_states_ > std::numeric_limits<state_id_t>::max()) {
            throw std::runtime_error("Number of states too high - revise source code and change state_id_t datatype");
        }

        if (num_transitions_ > std::numeric_limits<transition_id_t>::max()) {
            throw std::runtime_error("Number of states too high - revise source code and change transition_id_t datatype");
        }
    }

    ~TransitionSample() = default;
    TransitionSample(const TransitionSample&) = default;
    TransitionSample(TransitionSample&&) = default;

    std::size_t num_states() const { return num_states_; }
    std::size_t num_transitions() const { return num_transitions_; }

    inline unsigned get_transition_id(state_id_t s, state_id_t t) const { return transition_ids_.at(state_pair(s, t)); }
    inline const state_pair& get_state_pair(unsigned tx) const { return transition_ids_inv_.at(tx); }

    const transition_set_t& positive() const { return positive_transitions_; }
    const transition_set_t& negative() const { return negative_transitions_; }

    int vstar(unsigned sid) const {
        auto vstar = hstar_.at(sid);
        return vstar;
//        return vstar < 0 ? -1 : vstar;
    }

    const std::vector<unsigned>& successors(unsigned s) const {
        return trdata_.at(s);
    }

    bool is_expanded(unsigned state) const { return is_state_expanded_.at(state); }
    bool is_alive(unsigned state) const { return is_state_alive_.at(state); }
    bool is_goal(unsigned state) const { return is_state_goal_.at(state); }
    bool is_unsolvable(unsigned state) const { return is_state_unsolvable_.at(state); }

    unsigned num_unsolvable() const { return unsolvable_states_.size(); }

    const std::vector<unsigned>& all_alive() const { return alive_states_; }
    const std::vector<unsigned>& all_goals() const { return goal_states_; }

    //! Print a representation of the object to the given stream.
    friend std::ostream& operator<<(std::ostream &os, const TransitionSample& o) { return o.print(os); }
    std::ostream& print(std::ostream &os) const {
        os << "Transition sample [states: " << num_states_ << ", transitions: " << num_transitions_ << std::endl;
//        for (unsigned s = 0; s < num_states_; ++s) {
//            const auto& dsts = trdata_[s];
//            if (!dsts.empty()) os << "state " << s << ":";
//            for (auto dst:dsts) os << " " << dst;
//            os << std::endl;
//        }
        return os;
    }

    void read(std::istream &is) {
        // From here on, each line contains information about one transition in the sample, in the form of a tuple
        // (s, s', L),
        // where s and s' prime are the IDs of the states involved in the transition, and L is the classification label
        int s = -1, sprime = -1, label = -1;

        for (unsigned i = 0; i < num_transitions_; ++i) {
            is >> s >> sprime >> label;
            if (s < 0 || s >= num_states_ || sprime < 0 || sprime >= num_states_ || label < 0 || label > 1) {
                throw std::runtime_error("Wrong sample file format");
            }

            if (i != (unsigned) transition_ids_inv_.size()) throw std::runtime_error("Unexpected transition ID");

            transition_ids_inv_.emplace_back((state_id_t) s, (state_id_t) sprime);
            auto it1 = transition_ids_.emplace(transition_ids_inv_.back(), i);
            if (!it1.second) throw std::runtime_error("Duplicate transition");

            if (label == 1) positive_transitions_.insert(i);
            else negative_transitions_.insert(i);
        }
    }

    static TransitionSample read_dump(std::istream &is, bool verbose) {
        unsigned num_states = 0, num_transitions = 0;
        is >> num_states >> num_transitions;
        TransitionSample transitions(num_states, num_transitions);
        transitions.read(is);
        if( verbose ) {
            std::cout << "TransitionSample::read_dump: #states=" << transitions.num_states()
                      << ", #transitions=" << transitions.num_transitions()
                      << std::endl;
        }
        return transitions;
    }
};

} // namespaces

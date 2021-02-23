
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

    //! trdata_[s] contains the IDs of all neighbors of s in the state space
    std::vector<std::vector<unsigned>> trdata_;

    std::vector<bool> is_state_alive_;
    std::vector<bool> is_state_goal_;
    std::vector<bool> is_state_unsolvable_;

    std::vector<unsigned> alive_states_;
    std::vector<unsigned> goal_states_;

    std::vector<int> vstar_;

public:
    TransitionSample(std::size_t num_states, std::size_t num_transitions)
            : num_states_(num_states),
              num_transitions_(num_transitions),
              trdata_(num_states),
              is_state_alive_(num_states, false),
              is_state_goal_(num_states, false),
              is_state_unsolvable_(num_states, false),
              alive_states_(),
              goal_states_()
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

    int vstar(unsigned sid) const { return vstar_.at(sid); }

    const std::vector<unsigned>& successors(unsigned s) const {
        return trdata_.at(s);
    }

    bool is_alive(unsigned state) const { return is_state_alive_.at(state); }
    bool is_goal(unsigned state) const { return is_state_goal_.at(state); }
    bool is_unsolvable(unsigned state) const { return is_state_unsolvable_.at(state); }

    unsigned num_unsolvable() const {
        unsigned c = 0;
        for (unsigned s=0; s < num_states_; ++s) {
            if (is_unsolvable(s)) c += 1;
        }
        return c;
    }

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

    // readers
    void read(std::istream &is) {
        // read transitions, in format: source_id, num_successors, succ_1, succ_2, ...
        for (unsigned i = 0; i < num_states_; ++i) {
            unsigned src = 0, count = 0, dst = 0;
            is >> src >> count;
            assert(src < num_states_ && 0 <= count);
            if (count > 0) {
                std::vector<bool> seen(num_states_, false);
                trdata_[src].reserve(count);
                for (unsigned j = 0; j < count; ++j) {
                    is >> dst;
                    assert(dst < num_states_);
                    if (seen.at(dst)) throw std::runtime_error("Duplicate transition");
                    trdata_[src].push_back(dst);
                    seen[dst] = true;
                }
            }
        }

        // Store the value of V^*(s) for each state s
        unsigned s = 0;
        int vstar = 0;
        vstar_.reserve(num_states_);
        for (s=0; s < num_states_; ++s) {
            is >> vstar;
            vstar_.push_back(vstar);
            if (vstar>0) {
                is_state_alive_[s] = true;
                alive_states_.push_back(s);
            } else if (vstar<0) {
                is_state_unsolvable_[s] = true;
            } else {
                goal_states_.push_back(s);
                is_state_goal_[s] = true;
            }
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

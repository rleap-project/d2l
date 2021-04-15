
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

    std::vector<bool> is_state_expanded_;
    std::vector<bool> is_state_alive_;
    std::vector<bool> is_state_goal_;
    std::vector<bool> is_state_unsolvable_;
    std::vector<int> hstar_;

    std::vector<unsigned> expanded_states_;
    std::vector<unsigned> alive_states_;
    std::vector<unsigned> goal_states_;
    std::vector<unsigned> unsolvable_states_;

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
              expanded_states_(),
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
        // From here on, line #i contains a number of space-separated information fields about state #i.
        // The fields, in order, are:
        // <state-id>: The ID of the state, should correspond to the number of the line-1
        //             (because of the first line that once we are here has already been processed)
        // <expanded?>: Whether the state has been expanded in the sample or not
        // <goal?>: Whether the state is a goal
        // <unsolvable?>: Whether the state is unsolvable.
        // <h*(s)>: The optimal distance-to-goal, if known, or -2, if not. A value of -1 indicates infinity (i.e. s unsolvable).
        // <num_successors>: The number of out-transitions starting on the state. Can be 0 if state has not been
        //                    expanded, or of course if state has no successors in the problem
        // <s_i>: one state ID for each possible successor
        int sid = 0, nsuccessors = 0, dst = 0, hstar = 0;
        bool expanded = false, goal = false, unsolvable = false;

        for (unsigned i = 0; i < num_states_; ++i) {
            is >> sid >> expanded >> goal >> unsolvable >> hstar >> nsuccessors;
            assert(sid == i && 0 <= nsuccessors);

            is_state_expanded_[sid] = expanded;
            is_state_goal_[sid] = goal;
            is_state_unsolvable_[sid] = unsolvable;

            if (expanded) expanded_states_.push_back(sid);
            if (goal) goal_states_.push_back(sid);
            if (unsolvable) unsolvable_states_.push_back(sid);

//            std::cout << "hstar(" << sid << ")=" << hstar << std::endl;

            hstar_[sid] = hstar;
            if (hstar > 0) {
                is_state_alive_[sid] = true;
                alive_states_.push_back(sid);
            } else {
                // No need to do anything
            }

            if (nsuccessors > 0) {
                std::vector<bool> seen(num_states_, false);
                trdata_[sid].reserve(nsuccessors);
                for (unsigned j = 0; j < nsuccessors; ++j) {
                    is >> dst;
                    assert(dst < num_states_);
                    if (seen.at(dst)) throw std::runtime_error("Duplicate transition");
                    trdata_[sid].push_back(dst);
                    seen[dst] = true;
                }
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

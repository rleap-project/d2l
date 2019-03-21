
#pragma once

#include <blai/sample.h>

class CNFGenerator {
public:
    using D1_map_t = std::map<int, int>;
    using D2_map_t = std::map<std::pair<int, int>, int>;
    using D_t = std::vector<std::vector<int> >;

protected:
    const Sample::Sample& sample_;

public:
    CNFGenerator(const Sample::Sample& sample) :
        sample_(sample)
    {}


    // fill in given D1 and D2 data-structures
    size_t populate(D_t &D1, D1_map_t &D1_map, bool remove_dummy_features, bool complete_only_for_marked_transitions = false) const {
        size_t total_size = 0;
        for( int s = 0; s < sample_.matrix_->num_states(); ++s ) {
            for( int t = s + 1; t < sample_.matrix_->num_states(); ++t ) {
                if( !sample_.marked(s, complete_only_for_marked_transitions) && !sample_.marked(t, complete_only_for_marked_transitions) ) continue;
                int index = s * sample_.matrix_->num_states() + t;
                D1_map.insert(std::make_pair(index, D1.size()));
                D1.emplace_back();
                for( int f = 0; f < sample_.matrix_->num_features(); ++f ) {
                    int sf = (*sample_.matrix_)(s, f);
                    int tf = (*sample_.matrix_)(t, f);
                    if( ((sf == 0) && (tf > 0)) || ((sf > 0) && (tf == 0)) ) {
                        assert(!sample_.matrix_->is_dummy(f));
                        D1.back().push_back(remove_dummy_features ? sample_.matrix_->remap_feature(f) : f);
                    }
                }
                total_size += D1.back().size();
            }
        }
        return total_size;
    }
    size_t populate(std::vector<std::vector<int> > &D2, std::map<std::pair<int, int>, int> &D2_map, bool remove_dummy_features, bool complete_only_for_marked_transitions = false) const {
        size_t total_size = 0;
        for( int s = 0; s < sample_.matrix_->num_states(); ++s ) {
            if( !sample_.expanded(s) ) continue;

            std::vector<int> succ_s;
            sample_.transitions_->successors(s, succ_s);
            for( int t = s + 1; t < sample_.matrix_->num_states(); ++t ) {
                if( !sample_.expanded(t) ) continue;
                if( sample_.matrix_->goal(s) != sample_.matrix_->goal(t) ) continue;
                if( !sample_.marked(s, complete_only_for_marked_transitions) && !sample_.marked(t, complete_only_for_marked_transitions) ) continue;

                std::vector<int> succ_t;
                sample_.transitions_->successors(t, succ_t);
                for( int i = 0; i < int(succ_s.size()); ++i ) {
                    int s_next = succ_s[i];

                    int index_s = s * sample_.matrix_->num_states() + s_next;
                    for( int j = 0; j < int(succ_t.size()); ++j ) {
                        int t_next = succ_t[j];
                        if( !sample_.marked(s, s_next, complete_only_for_marked_transitions) && !sample_.marked(t, t_next, complete_only_for_marked_transitions) ) continue;

                        int index_t = t * sample_.matrix_->num_states() + t_next;
                        D2_map.insert(std::make_pair(std::make_pair(index_s, index_t), D2.size()));
                        D2.emplace_back();
                        for( int f = 0; f < sample_.matrix_->num_features(); ++f ) {
                            int spf = (*sample_.matrix_)(succ_s[i], f);
                            int sf = (*sample_.matrix_)(s, f);
                            int type_s = spf - sf; // <0 if DEC, =0 if unaffected, >0 if INC
                            type_s = type_s < 0 ? -1 : (type_s > 0 ? 1 : 0);

                            int tpf = (*sample_.matrix_)(succ_t[j], f);
                            int tf = (*sample_.matrix_)(t, f);
                            int type_t = tpf - tf; // <0 if DEC, =0 if unaffected, >0 if INC
                            type_t = type_t < 0 ? -1 : (type_t > 0 ? 1 : 0);

                            if( type_s != type_t ) {
                                assert(!sample_.matrix_->is_dummy(f));
                                D2.back().push_back(remove_dummy_features ? sample_.matrix_->remap_feature(f) : f);
                            }
                        }
                        total_size += D2.back().size();
                    }
                }
            }
        }
        return total_size;
    }
    
    bool dump_bridge(std::ostream &os, const D1_map_t &D1_map, const D2_map_t &D2_map, const D_t &D1, const D_t &D2, int s, int s_next, int t, bool verbose) const {

        assert(sample_.matrix_->goal(s) == sample_.matrix_->goal(t));
        assert(!sample_.expanded(s));
        assert(!sample_.expanded(t));

        // order states s and t
        int small = s < t ? s : t;
        int large = s < t ? t : s;
        assert(small < large);

        // define index for (s,s') and get successors of t
        int index_s = s * sample_.matrix_->num_states() + s_next;
        std::vector<int> succ_t;
        sample_.transitions_->successors(t, succ_t);

        // -D1(s,t) => OR_{t'} -D2(s,s',t,t')  for transition (s,s')

        // antecedent: -D1(s,t)
        int index_st = small * sample_.matrix_->num_states() + large;
        D1_map_t::const_iterator it = D1_map.find(index_st);
        assert(it != D1_map.end());
        int k = it->second;
        os << D1[k].size();
        for( int j = 0; j < int(D1[k].size()); ++j )
            os << " " << -(1 + D1[k][j]);

        // consequent: OR_{t'} -D2(s,s',t,t')
        os << " " << sample_.transitions_->num_transitions(t);
        for( int j = 0; j < int(succ_t.size()); ++j ) {
            int index_t = t * sample_.matrix_->num_states() + succ_t[j];
            D2_map_t::const_iterator jt = D2_map.find(std::make_pair(index_s, index_t));
            assert(jt != D2_map.end());
            int k = jt->second;
            os << " " << D2[k].size();
            for( int l = 0; l < int(D2[k].size()); ++l )
                os << " " << -(1 + D2[k][l]);
        }
        os << std::endl;

        // generate warning if D1(s,t) is empty AND t has no successors
        if( D1[k].empty() && (sample_.transitions_->num_transitions(t) == 0) ) {
            if( verbose ) {
                std::cout << Utils::warning()
                          << "theory is UNSAT:"
                          << " D1(s=" << s << ",t=" << t << ") is empty,"
                          << " there is transition (s=" << s << ",s'=" << s_next << "),"
                          << " there are no transitions for t=" << t
                          << std::endl;
            }
            return false;
        }
        return true;
    }
    void dump_target(std::ostream &os, const D1_map_t &D1_map, const D_t &D1, int s, int t, bool verbose) const {
        int index = s * sample_.matrix_->num_states() + t;
        D1_map_t::const_iterator it = D1_map.find(index);
        assert(it != D1_map.end());
        int k = it->second;
        assert(!D1[k].empty());
        os << D1[k].size();
        for( int i = 0; i < int(D1[k].size()); ++i )
            os << " " << 1 + D1[k][i];
        os << std::endl;
    }

    bool dump_weighted_max_sat_problem(std::ostream &os, bool verbose) const {
        assert(sample_.matrix_ != nullptr);
        assert(sample_.transitions_ != nullptr);

        int num_states = sample_.matrix_->num_states();
        int num_features = sample_.matrix_->num_features();
        int num_dummy_features = sample_.matrix_->num_dummy_features();
        assert(num_features > num_dummy_features);

        // populate D1: D1(s,t) contains the features that make s and t different (s < t)
        D_t D1;
        D1_map_t D1_map;
        populate(D1, D1_map, true);

        // populate D2: D2(s,s',t,t') contains the features that make (s,s') and (t,t') different (s < t)
        D_t D2;
        D2_map_t D2_map;
        populate(D2, D2_map, true);

        // count bridge formulas
        int num_formulas = 0;
        for( int s = 0; s < num_states; ++s ) {
            if( !sample_.expanded(s) ) continue;
            for( int t = s + 1; t < num_states; ++t ) {
                if( (sample_.matrix_->goal(s) == sample_.matrix_->goal(t)) && sample_.expanded(t) )
                    num_formulas += sample_.transitions_->num_transitions(s);
                num_formulas += sample_.transitions_->num_transitions(t);
            }
        }

        // count targets
        int num_targets = 0;
        for( int s = 0; s < num_states; ++s ) {
            for( int t = s + 1; t < num_states; ++t ) {
                if( sample_.matrix_->goal(s) != sample_.matrix_->goal(t) ) {
                    int index = s * num_states + t;
                    assert(D1_map.find(index) != D1_map.end());
                    int k = D1_map[index];
                    if( D1[k].empty() ) {
                        if( verbose ) {
                            std::cout << "UNSAT: D1(s=" << s << ",t=" << t << ") is empty:"
                                      << " s.goal()=" << sample_.matrix_->goal(s) << ","
                                      << " t.goal()=" << sample_.matrix_->goal(t)
                                      << std::endl;
                        }
                        return false;
                    }
                    ++num_targets;
                }
            }
        }

        // dump Weighted MaxSAT problem
        os << num_features - num_dummy_features << " " << num_formulas << " " << num_targets << std::endl;

        // dump feature costs
        os << num_features - num_dummy_features;
        for( int i = 0; i < num_features; ++i ) {
            if( !sample_.matrix_->is_dummy(i) ) {
                assert(sample_.matrix_->feature_cost(i) > 0);
                os << " " << sample_.matrix_->feature_cost(i);
            }
        }
        os << std::endl;

        // dump bridge formulas
        for( int s = 0; s < num_states; ++s ) {
            if( !sample_.expanded(s) ) continue;
            std::vector<int> succ_s;
            sample_.transitions_->successors(s, succ_s);
            for( int i = 0; i < int(succ_s.size()); ++i ) {
                int s_next = succ_s[i];
                for( int t = 0; t < num_states; ++t ) {
                    if( s == t ) continue;
                    if( !sample_.expanded(t) ) continue;
                    if( sample_.matrix_->goal(s) != sample_.matrix_->goal(t) ) continue;
                    if( !dump_bridge(os, D1_map, D2_map, D1, D2, s, s_next, t, verbose) )
                        return false;
                }
            }
        }

#if 0
        // dump bridge formulas
        for( int s = 0; s < num_states; ++s ) {
            if( !sample_.expanded(s) ) continue;
            std::vector<int> succ_s;
            sample_.transitions_->successors(s, succ_s);
            for( int i = 0; i < int(succ_s.size()); ++i ) {
                int index_s = s * num_states + succ_s[i];
                for( int t = s + 1; t < num_states; ++t ) {
                    if( sample_.matrix_->goal(s) != sample_.matrix_->goal(t) ) continue;
                    if( !sample_.expanded(t) ) continue;

                    std::vector<int> succ_t;
                    sample_.transitions_->successors(t, succ_t);
                    assert(int(succ_t.size()) == sample_.transitions_->num_transitions(t));

                    // implications: (1) -D1(s,t) => OR_{t'} -D2(s,s',t,t')  for transition (s,s')
                    //               (1) -D1(s,t) => OR_{s'} -D2(s,s',t,t')  for transition (t,t')

                    // antecedent for (1): -D1(s,t)
                    int index_st = s * num_states + t;
                    assert(D1_map.find(index_st) != D1_map.end());
                    int k = D1_map[index_st];
                    os << D1[k].size();
                    for( int j = 0; j < int(D1[k].size()); ++j )
                        os << " " << -(1 + D1[k][j]);

                    // consequent for (1): OR_{t'} -D2(s,s',t,t')
                    os << " " << sample_.transitions_->num_transitions(t);
                    for( int j = 0; j < int(succ_t.size()); ++j ) {
                        int index_t = t * num_states + succ_t[j];
                        assert(D2_map.find(std::make_pair(index_s, index_t)) != D2_map.end());
                        int k = D2_map[std::make_pair(index_s, index_t)];
                        os << " " << D2[k].size();
                        for( int l = 0; l < int(D2[k].size()); ++l )
                            os << " " << -(1 + D2[k][l]);
                    }
                    os << std::endl;

                    // generate warning if D1(s,t) is empty AND t has no successors
                    if( D1[k].empty() && (sample_.transitions_->num_transitions(t) == 0) ) {
                        if( verbose ) {
                            std::cout << Utils::warning()
                                      << "theory is UNSAT:"
                                      << " D1(s=" << s << ",t=" << t << ") is empty,"
                                      << " there is transition (s=" << s << ",s'=" << succ_s[i] << "),"
                                      << " there are no transitions for t=" << t
                                      << std::endl;
                        }
                        return false;
                    }
                }
            }
        }
#endif

        // dump targets
        for( int s = 0; s < num_states; ++s ) {
            for( int t = s + 1; t < num_states; ++t ) {
                if( sample_.matrix_->goal(s) != sample_.matrix_->goal(t) )
                    dump_target(os, D1_map, D1, s, t, verbose);
            }
        }

        return true;
    }

};
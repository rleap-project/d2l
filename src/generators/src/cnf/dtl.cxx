//
// Created by js on 15/3/21.
//


#include "dtl.h"
#include "types.h"

#include <iostream>
#include <vector>
#include <unordered_map>

#include <boost/functional/hash.hpp>
#include <common/helpers.h>


namespace sltp::cnf {

    std::pair<cnf::CNFGenerationOutput, VariableMapping> DTLEncoding::generate(CNFWriter& wr)
    {
        using Wr = CNFWriter;
        const auto& mat = sample_.matrix();
        const auto num_transitions = transition_ids_.size();

        VariableMapping variables(nf_);

        // We keep the logging of variables for debugging purposes
        auto varmapstream = utils::get_ofstream(options.workspace + "/varmap.wsat");

        /////// CNF parameters ///////
        const unsigned m = options.n_features;
        //const unsigned K = m; // maximum levels of induction when proving non-termination
        const unsigned K = options.consistency_bound; // V*(s) <= K must be consistent, i.e. Good(s,s') -> V(s') < V(s)
        const unsigned max_d = compute_D();
        const unsigned n_constraints = 8; // fixed by the encoding

        std::unordered_set< sltp::state_id_t > s_full;
        /////// END CNF parameters ///////

        /////// CNF variables ///////
        // Keep a map from each feature index and i bit position to the SAT variable ID of Selected(f,i)
        std::vector<cnfvar_t> selecteds_pos(nf_ * m, std::numeric_limits<uint32_t>::max());

        // Vector for cardinality constraints of selected features
        unsigned n_card_constraints = 0;
        for( unsigned aux = feature_ids.size(); aux > 0; aux>>=1, n_card_constraints++ ){}
        std::vector<cnfvar_t> card_constraint_vars( m * n_card_constraints, std::numeric_limits<uint32_t>::max() );

        // Keep a map from each i bit position and state to the SAT variable ID of b_i(s)
        std::map< std::pair< unsigned, state_id_t >, cnfvar_t > b_vars;

        // Inc(i,s,s') and Dec(i,s,s') vars
        std::map< std::vector< unsigned >, cnfvar_t > inc_tx;
        std::map< std::vector< unsigned >, cnfvar_t > dec_tx;

        // C5 vars
        std::map< std::pair< unsigned, unsigned >, cnfvar_t > bprec_vars;
        std::map< std::pair< unsigned, unsigned >, cnfvar_t > beff_vars;
        std::map< std::pair< unsigned, unsigned >, cnfvar_t > aux_prec_1_vars;
        std::map< std::pair< unsigned, unsigned >, cnfvar_t > aux_prec_2_vars;

        // V vars
        std::map< std::pair< unsigned, unsigned >, cnfvar_t > v_vars;


        unsigned n_select_vars = 0;
        unsigned n_select_pos_vars = 0;
        unsigned n_y_vars = 0;
        unsigned n_tx_vars = 0;
        unsigned n_good_vars = 0;
        unsigned n_bi_vars = 0;
        unsigned n_split_vars = 0;
        unsigned n_v_vars = 0;

        std::vector< unsigned > cl_counter(n_constraints,0);

        // Create one "Select(f)" variable for each feature f in the pool
        for (unsigned f:feature_ids) {
            auto v = wr.var("Select(" + mat.feature_name(f) + ")");
            variables.selecteds[f] = v;
            varmapstream << f << "\t" << v << "\t" << mat.feature_name(f) << std::endl;
            ++n_select_vars;

            // Create one "Select(f,i)" variable for each feature f in the pool and position i in [0,m)
            for (unsigned i=0; i<m; i++ ){
                auto v2 = wr.var( "Select(" + mat.feature_name(f) +","+ std::to_string(i) + ")" );
                selecteds_pos[nf_*i + f] = v2;
                //selected_pos_map_stream << f << "\t" << i << "\t" << v2 << "\t" << mat.feature_name(f) << std::endl;
                varmapstream << f << "\t" << i << "\t" << v2 << "\t" << mat.feature_name(f) << std::endl;
                ++n_select_pos_vars;
            }
        }

        // Create log( F ) cardinality constraint vars
        for( unsigned i = 0; i < m; i++ ) {
            for (unsigned y = 0; y < n_card_constraints; y++) {
                auto var = wr.var("CardConstraint(" + std::to_string(i) + "," + std::to_string(y) + ")");
                card_constraint_vars[n_card_constraints*i + y] = var;
                varmapstream << i << " " << y << " " << var << std::endl;
                ++n_y_vars;
            }
        }

        // Create a variable "Good(s, s')" for each transition (s, s') such that s' is solvable and (s, s') is not in BAD
        for( const auto s : sample_.alive_states()){
            for( const auto s_prime : sample_.successors(s) ){
                auto tx = get_transition_id(s,s_prime);
                auto repr = get_representative_id(tx);

                if( tx != repr ) continue;

                cnfvar_t good_s_sprime = wr.var("Good(" + std::to_string(s) + ", " + std::to_string(s_prime) + ")");
                auto it = variables.goods.emplace(tx, good_s_sprime);
                assert(it.second); // i.e. the SAT variable Good(s, s') is necessarily new
                varmapstream << good_s_sprime << " " << s << " " << s_prime << std::endl;
                n_good_vars++;

                s_full.insert(s);
                s_full.insert(s_prime);

                for( unsigned i = 0; i < m; i++ ){
                    cnfvar_t inc_tx_var = wr.var( "Inc(" + std::to_string(i) + "," + std::to_string(s) + "," + std::to_string(s_prime) +")" );
                    inc_tx.emplace( std::vector<unsigned>({i,s,s_prime}), inc_tx_var );
                    varmapstream << inc_tx_var << " " << i << " " << s << " " << s_prime << std::endl;

                    cnfvar_t dec_tx_var = wr.var( "Dec(" + std::to_string(i) + "," + std::to_string(s) + "," + std::to_string(s_prime) + ")" );
                    dec_tx.emplace( std::vector<unsigned>({i,s,s_prime}), dec_tx_var );
                    varmapstream << dec_tx_var << " " << i << " " << s << " " << s_prime << std::endl;

                    n_tx_vars += 2;
                }
            }
        }

        // Create one variable "b_i(s)" denoting the truth value of the i-th bit in [0,m) of the abstraction of s
        for( const auto s : s_full ){
            for( unsigned i = 0; i < m; i++ ){
                cnfvar_t b_var = wr.var( "b_" +std::to_string(i) + "(" + std::to_string(s) + ")" );
                b_vars.emplace( std::make_pair(i,s), b_var );
                varmapstream << b_var << " " << i << " " << s << std::endl;
                ++n_bi_vars;
            }
        }

        // Cardinality Networks over F: at-most-m features
        // Vars + clauses
        /*int prefix_id = 0;
        std::vector<cnfvar_t > yvars;
        sorting_network(wr,prefix_id,variables.selecteds,yvars);
        for( int i = m; i < int(yvars.size()); ++i ){
            wr.cl({
                wr.lit(yvars[i],false)
            });
        }
        n_select_vars += prefix_id;*/

        for( unsigned i = 0; i + 1 < m ; i++){
            for( unsigned j = i+1; j<m; j++ ){
                cnfvar_t bprec_var = wr.var( "BPrec(" + std::to_string(i) + "," + std::to_string(j) + ")");
                bprec_vars.emplace( std::make_pair(i,j), bprec_var );
                varmapstream << bprec_var << " " << i << " " << j << std::endl;

                cnfvar_t beff_var = wr.var( "BEff(" + std::to_string(i) + "," + std::to_string(j) + ")");
                beff_vars.emplace( std::make_pair(i,j), beff_var );
                varmapstream << beff_var << " " << i << " " << j << std::endl;

                cnfvar_t aux_prec_1_var = wr.var( "AuxPrec1(" + std::to_string(i) + "," + std::to_string(j) + ")");
                aux_prec_1_vars.emplace( std::make_pair(i,j), aux_prec_1_var );
                varmapstream << aux_prec_1_var << " " << i << " " << j << std::endl;

                cnfvar_t aux_prec_2_var = wr.var( "AuxPrec2(" + std::to_string(i) + "," + std::to_string(j) + ")");
                aux_prec_2_vars.emplace( std::make_pair(i,j), aux_prec_2_var );
                varmapstream << aux_prec_2_var << " " << i << " " << j << std::endl;

                n_split_vars+=4;
            }
        }

        /*for( const auto s : sample_.alive_states()){
            //int min_vs = get_vstar(s);
            //int max_vs = get_max_v(s);
            for( unsigned d = 1; d <= max_d; d++ ) {
                const auto v_s_d = wr.var("V(" + std::to_string(s) + ", " + std::to_string(d) + ")");
                v_vars.emplace(std::make_pair(s, d), v_s_d);
                n_v_vars++;
                varmapstream << v_s_d << " " << s << " " << d << std::endl;
            }
        }*/

        // Create variables V(s, d) variables for all alive state s and d in 1..D
        for (const auto s:sample_.alive_states()) {
            const auto min_vs = get_vstar(s);
            const auto max_vs = get_max_v(s);
            assert(max_vs > 0 && max_vs <= max_d && min_vs <= max_vs);

            cnfclause_t within_range_clause;

            // TODO Note that we could be more efficient here and create only variables V(s,d) for those values of d that
            //  are within the bounds below. I'm leaving that as a future optimization, as it slightly complicates the
            //  formulation of constraints C2
            for (unsigned d = 1; d <= max_d; ++d) {
                const auto v_s_d = wr.var("V(" + std::to_string(s) + ", " + std::to_string(d) + ")");
                v_vars.emplace(std::make_pair(s, d), v_s_d);
//                std::cout << s << ", " << d << std::endl;
                n_v_vars++;
                if (d >= min_vs && d <= max_vs) {
                    within_range_clause.push_back(Wr::lit(v_s_d, true));
                }
            }

            if( min_vs > K ) continue; // Possible bug

            // Add clauses (4), (5)
            wr.cl(within_range_clause);
            cl_counter[6] += 1;

            for (unsigned d = 1; d <= max_d; ++d) {
                for (unsigned dprime = d+1; dprime <= max_d; ++dprime) {
                    wr.cl({Wr::lit(v_vars.at({s, d}), false), Wr::lit(v_vars.at({s, dprime}), false)});
                    cl_counter[6] += 1;
                }
            }
        }

        // Check our variable count is correct

        assert(wr.nvars() == n_select_vars + n_select_pos_vars + n_y_vars + n_good_vars + n_bi_vars + n_tx_vars
                             + n_split_vars + n_v_vars );

        // From this point on, no more variables will be created. Print total count.
        std::cout << "A total of " << wr.nvars() << " variables were created" << std::endl;
        std::cout << "\tSelect(f): " << n_select_vars << std::endl;
        std::cout << "\tSelect(f,i): " << n_select_pos_vars << std::endl;
        std::cout << "\tCardConstraint(i,y): " << n_y_vars << std::endl;
        std::cout << "\tGood(s, s'): " << n_good_vars << std::endl;
        std::cout << "\tb_i(s): " << n_bi_vars << std::endl;
        std::cout << "\tInc(i,s,s'), Dec(i,s,s'): " << n_tx_vars << std::endl;
        std::cout << "\tBPrec(i,j), BEff(i,j): " << n_split_vars << std::endl;
        std::cout << "\tV(s,d): " << n_v_vars << std::endl;
        /////// END CNF variables ///////

        /////// CNF clauses ///////
        // C1. Choose exactly m features, breaking symmetries
        if (options.verbosity>0) std::cout << "Encoding clause C1..." << std::endl;
        for (unsigned i=0; i < m; ++i) {
            // C1.a
            cnfclause_t clause;
            for (const auto f:feature_ids) {
                clause.push_back(wr.lit(selecteds_pos.at(nf_*i + f), true));
            }
            wr.cl(clause);
            cl_counter[1]++;

            // C1.b
            for( const auto f:feature_ids ){
                // C1.b Select(f) -> OR_i Select(f,i)
                cnfclause_t clause_select;
                clause_select.push_back( wr.lit(variables.selecteds.at( f ), false ) );
                for( unsigned i = 0; i < m; i++ ){
                    // C1.b Select(f) <- OR_i Select(f,i)
                    wr.cl( {
                                   wr.lit( selecteds_pos.at( nf_ * i + f ), false ),
                                   wr.lit( variables.selecteds.at( f ), true  )
                           } );
                    cl_counter[1]++;
                    clause_select.push_back( wr.lit( selecteds_pos.at( nf_ * i + f ), true ) );
                }
                wr.cl( clause_select );
                cl_counter[1]++;
            }
        }

        // C1.c |F|^2
        /*for (unsigned f=0; f < nf_; ++f) {
            for (unsigned f_prime=f+1; f_prime < nf_; ++f_prime) {
                wr.cl({
                    wr.lit(selecteds_pos.at(nf_*i + f), false),
                    wr.lit(selecteds_pos.at(nf_*i + f_prime), false)
                });
                n_selected_clauses++;
            }
        }*/
        // C1.c |F|log(|F|) AMO(i,f)
        for (unsigned i=0; i < m; ++i) {
            for( unsigned f = 0; f < nf_; ++f ){
                for( unsigned y = 0; y < n_card_constraints; y++ ){
                    bool eval = ( (f & (1 << y ) ) > 0 );
                    wr.cl({
                                  wr.lit( selecteds_pos.at( nf_*i + f ), false ),
                                  wr.lit( card_constraint_vars[ n_card_constraints*i + y ], eval )
                          });
                    cl_counter[1]++;
                }
            }
        }

        // C2 Distinguish good transitions and goals from dead-end transitions
        if (options.verbosity>0) std::cout << "Encoding clause C2..." << std::endl;

        // C2.a Force D1(s1, s2) to be true if exactly one of the two states is a goal state
        std::set< cnfclause_t > aux_c2;
        for ( const auto s : sample_.full_training_set().all_alive()) {
            for( const auto t : sample_.full_training_set().all_goals()) {
                const auto d1feats = compute_d1_distinguishing_features(sample_, s, t);
                if (d1feats.empty()) {
                    undist_goal_warning(s, t);
                }

                cnfclause_t clause;
                for (unsigned f:d1feats) {
                    clause.push_back(Wr::lit(variables.selecteds.at(f), true));
                }
                aux_c2.emplace( clause );
            }
        }

        for( const auto goal_d1_clauses : aux_c2 ){
            wr.cl( goal_d1_clauses );
            cl_counter[2]++;
        }
        aux_c2.clear();

        /*for( const auto t : sample_.alive_states()){
            for( const auto t_prime : sample_.successors(t) ){
                if( sample_.is_solvable(t_prime) ) continue;
                const auto t_tx = get_transition_id(t,t_prime);
                const auto t_repr = get_representative_id( t_tx );

                if( t_tx != t_repr ) continue;

                for( const auto s : sample_.alive_states()){
                    for (const auto s_prime : sample_.successors(s)) {
                        if( !sample_.is_solvable(s_prime) ) continue;
                        const auto s_tx = get_transition_id(s,s_prime);
                        const auto s_repr = get_representative_id( s_tx );

                        if( s_tx != s_repr ) continue;

                        cnfclause_t clause;
                        clause.push_back( wr.lit( variables.goods.at(s_repr), false ) );
                        clause.push_back( wr.lit( variables.goods.at(t_repr), true ) );

                        for (feature_t f:compute_d1d2_distinguishing_features(feature_ids, sample_, s, s_prime, t, t_prime)) {
                            clause.push_back(Wr::lit(variables.selecteds.at(f), true));
                        }

                        wr.cl( clause );
                        cl_counter[2]++;
                    }
                }
                // C2.c -Good(t,t') for t solvable and t' unsolvable
                wr.cl( {wr.lit( variables.goods.at( t_repr ), false )} );
                cl_counter[2]++;
            }
        }*/

        // C2.b Good(s,s') & Good(t,t') -> OR_{f in D1&2(s,s',t,t')} Select(f);
        auto transitions_to_distinguish = distinguish_all_transitions();

        for (const auto& tpair:transitions_to_distinguish) {
            assert (!is_necessarily_bad(tpair.tx1));
            const auto& [s, sprime] = get_state_pair(tpair.tx1);
            const auto& [t, tprime] = get_state_pair(tpair.tx2);

            cnfclause_t clause{Wr::lit(variables.goods.at(tpair.tx1), false)};

            // Compute first the Selected(f) terms
            for (feature_t f:compute_d1d2_distinguishing_features(feature_ids, sample_, s, sprime, t, tprime)) {
                clause.push_back(Wr::lit(variables.selecteds.at(f), true));
            }

            if (!is_necessarily_bad(tpair.tx2)) {
                auto good_t_tprime = variables.goods.at(tpair.tx2);
                clause.push_back(Wr::lit(good_t_tprime, true));
            }
            wr.cl(clause);
            cl_counter[2]++;
        }

        // C2.c  !Good(t,t') for t alive, t' deadend
        std::set< unsigned > bad_tx;
        for( const auto t : sample_.alive_states()) {
            for (const auto t_prime : sample_.successors(t)) {
                if (sample_.is_solvable(t_prime)) continue;
                const auto t_tx = get_transition_id(t,t_prime);
                const auto t_repr = get_representative_id( t_tx );
                bad_tx.insert( t_repr );
            }
        }
        for( const auto repr : bad_tx ){
            wr.cl( {wr.lit(variables.goods.at( repr),false)} );
            cl_counter[2]++;
        }
        bad_tx.clear();

        // C3.b -Good(s,s') iff s is alive and s' is a dead-end
        /*std::set< unsigned > c2_repr;
        for( const auto s : sample_.alive_states() ){
            for( const auto s_prime : sample_.successors(s) ){
                //if( !sample_.is_unsolvable(s_prime) ) continue;
                const auto tx = get_transition_id(s,s_prime);
                if(!is_necessarily_bad(tx) ) continue;
                const auto repr = get_representative_id(tx);
                c2_repr.insert( repr );
            }
        }
        for( const auto repr : c2_repr ){
            wr.cl({ wr.lit(variables.goods.at(repr ), false ) });
        }*/
        //c2_repr.clear();

        // C3. For each state s, post a constraint OR_{s' child of s} Good(s, s')
        if (options.verbosity>0) std::cout << "Encoding clause C3..." << std::endl;
        for( const auto s : sample_.alive_states()){
            std::set< unsigned > c3_repr;
            for (unsigned s_prime:sample_.successors(s)) {
                if( !sample_.is_solvable(s_prime) ) continue;
                auto tx = get_transition_id(s, s_prime);
                auto repr = get_representative_id( tx );
                c3_repr.insert( repr );
            }
            cnfclause_t clause;
            for( auto repr : c3_repr ){
                clause.push_back( Wr::lit(variables.goods.at(repr ), true ) );
            }
            if (clause.empty()) {
                throw std::runtime_error("State #" + std::to_string(s) + " has no successor that can be good");
            }
            wr.cl( clause );
            ++cl_counter[3];
        }

        // C4. Policy Termination Preliminaries
        if (options.verbosity>0) std::cout << "Encoding clause C4..." << std::endl;
        // C4.a Good(s,s') -> OR_f Select(f)
        for( const auto s : sample_.alive_states()){
            for( const auto s_prime : sample_.successors( s ) ) {
                if( !sample_.is_solvable(s_prime) ) continue;
                auto tx = get_transition_id(s, s_prime);
                auto repr = get_representative_id(tx);

                if( tx != repr ) continue;

                cnfclause_t good_tx_clause;
                good_tx_clause.push_back( wr.lit(variables.goods.at(repr ), false) );
                for( const auto f : feature_ids ) {
                    if( mat.entry(s_prime, f) != mat.entry(s, f) ) {
                        good_tx_clause.push_back( wr.lit(variables.selecteds.at(f), true) );
                    }
                }
                wr.cl(good_tx_clause);
                ++cl_counter[4];
            }
        }

        // C4.b
        for( unsigned i = 0; i < m; i++ ){
            for( const auto s : sample_.alive_states()){
                for (const auto s_prime : sample_.successors(s)) {
                    if( !sample_.is_solvable(s_prime) ) continue;
                    cnfclause_t incs_clause, decs_clause, unchanges_clause;

                    auto tx = get_transition_id( s, s_prime );
                    auto repr = get_representative_id( tx );

                    if( tx != repr ) continue;

                    incs_clause.push_back( wr.lit(variables.goods.at(repr ), false ) );
                    incs_clause.push_back( wr.lit( inc_tx.at({i,s,s_prime}), false ) );
                    decs_clause.push_back( wr.lit(variables.goods.at(repr ), false ) );
                    decs_clause.push_back( wr.lit( dec_tx.at({i,s,s_prime}), false ) );
                    unchanges_clause.push_back( wr.lit(variables.goods.at(repr ), false ) );
                    unchanges_clause.push_back( wr.lit( inc_tx.at({i,s,s_prime}), true ) );
                    unchanges_clause.push_back( wr.lit( dec_tx.at({i,s,s_prime}), true ) );

                    for( const auto f : feature_ids ) {
                        unsigned sf = mat.entry(s, f);
                        unsigned spf = mat.entry(s_prime, f);
                        if (spf > sf) { // incs
                            incs_clause.push_back( wr.lit(selecteds_pos.at(nf_ * i + f), true) );
                        } else if (spf < sf) { // decs
                            decs_clause.push_back( wr.lit(selecteds_pos.at(nf_ * i + f), true) );

                        } else { // unchange
                            unchanges_clause.push_back( wr.lit(selecteds_pos.at(nf_ * i + f), true) );
                        }
                    }
                    wr.cl( incs_clause );
                    wr.cl( decs_clause );
                    wr.cl( unchanges_clause );
                    cl_counter[4] += 3;
                }
            }
        }

        // C4.c
        for( const auto s : s_full ){
            for( unsigned i = 0; i < m; i++ ){
                cnfclause_t pos_clause, neg_clause;
                pos_clause.push_back( wr.lit( b_vars.at( std::make_pair(i,s) ), false ) );
                neg_clause.push_back( wr.lit( b_vars.at( std::make_pair(i,s) ), true ) );
                for( const auto f : feature_ids ){
                    if( mat.entry(s,f) > 0 ){
                        pos_clause.push_back( wr.lit( selecteds_pos.at( nf_ * i + f ), true ) );
                    }
                    else{
                        neg_clause.push_back( wr.lit( selecteds_pos.at( nf_ * i + f ), true ) );
                    }
                }
                wr.cl( pos_clause );
                wr.cl( neg_clause );
                cl_counter[4] += 2;
            }
        }

        // C4.d & C4.e
        for( unsigned i = 0; i+1 < m; i++ ) {
            for( unsigned j = i+1; j<m;j++) {
                cnfvar_t bprec = bprec_vars.at( std::make_pair(i,j) );
                cnfvar_t aux_prec_1 = aux_prec_1_vars.at( std::make_pair(i,j) );
                cnfvar_t aux_prec_2 = aux_prec_2_vars.at( std::make_pair(i,j) );
                cnfvar_t beff = beff_vars.at( std::make_pair(i,j) );

                // BPrec(i,j) <-> AuxPrec1(i,j) or AuxPrec2(i,j)
                wr.cl({
                              wr.lit( bprec, false ),
                              wr.lit( aux_prec_1, true ),
                              wr.lit( aux_prec_2, true )
                      });
                wr.cl({
                              wr.lit( bprec, true ),
                              wr.lit( aux_prec_1, false )
                      });
                wr.cl({
                              wr.lit( bprec, true ),
                              wr.lit( aux_prec_2, false )
                      });
                cl_counter[4] += 3;

                for( const auto s : sample_.alive_states()){
                    for( const auto s_prime : sample_.successors(s) ){
                        if( !sample_.is_solvable(s_prime) ) continue;
                        auto tx = get_transition_id(s, s_prime);
                        auto repr = get_representative_id(tx);

                        if( tx != repr ) continue;

                        // AuxPrec1 -> [Good(s,s') & Inc(i,s,s') -> !b_j(s)]
                        wr.cl({
                                      wr.lit( aux_prec_1, false ),
                                      wr.lit(variables.goods.at(repr), false),
                                      wr.lit(inc_tx.at({i, s, s_prime}), false),
                                      wr.lit( b_vars.at(std::make_pair(j,s)), false )
                              });
                        // AuxPrec1 -> [Good(s,s') & Dec(i,s,s') -> b_j(s)]
                        wr.cl({
                                      wr.lit( aux_prec_1, false ),
                                      wr.lit(variables.goods.at(repr), false),
                                      wr.lit(dec_tx.at({i, s, s_prime}), false),
                                      wr.lit( b_vars.at(std::make_pair(j,s)), true )
                              });
                        // AuxPrec2 -> [Good(s,s') & Inc(i,s,s') -> b_j(s)]
                        wr.cl({
                                      wr.lit( aux_prec_2, false ),
                                      wr.lit(variables.goods.at(repr), false),
                                      wr.lit(inc_tx.at({i, s, s_prime}), false),
                                      wr.lit( b_vars.at(std::make_pair(j,s)), true )
                              });
                        // AuxPrec2 -> [Good(s,s') & Dec(i,s,s') -> !b_j(s)]
                        wr.cl({
                                      wr.lit( aux_prec_2, false ),
                                      wr.lit(variables.goods.at(repr), false),
                                      wr.lit(dec_tx.at({i, s, s_prime}), false),
                                      wr.lit( b_vars.at(std::make_pair(j,s)), false )
                              });
                        cl_counter[4] += 4;

                        // BEff(i,j) -> AND_{s,s'} [Good(s,s') & Inc(i,s,s') -> Dec(j,s,s')]
                        wr.cl({
                                      wr.lit( beff, false ),
                                      wr.lit(variables.goods.at(repr), false),
                                      wr.lit(inc_tx.at({i, s, s_prime}), false),
                                      wr.lit( dec_tx.at({j,s,s_prime}), true )
                              });
                        cl_counter[4]++;
                    }
                }
            }
        }

        // C5. Policy termination
        if (options.verbosity>0) std::cout << "Encoding clause C5..." << std::endl;

        // C5.a Good(s,s') -> -Inc(m-1,s,s')
        for( const auto s : sample_.alive_states()){
            for( const auto s_prime : sample_.successors(s) ){
                if( !sample_.is_solvable(s_prime) ) continue;
                auto tx = get_transition_id(s,s_prime);
                auto repr = get_representative_id(tx);

                if( tx != repr ) continue;

                wr.cl({
                              wr.lit(variables.goods.at(repr), false),
                              wr.lit(inc_tx.at({m-1, s, s_prime}), false)
                      });
                cl_counter[5]++;
            }
        }

        // C5.b OR_{j>i} BPrec(i,j) or BEff(i,j)
        for( unsigned i =0; i + 1 < m; i++ ){
            cnfclause_t clause;
            for( unsigned j = i+1; j < m; j++ ){
                clause.push_back( wr.lit( bprec_vars.at( std::make_pair(i,j) ), true ) );
                clause.push_back( wr.lit( beff_vars.at( std::make_pair(i,j) ), true ) );
            }
            wr.cl( clause );
            cl_counter[5]++;
        }

        // C6 V^*(s) <= V(s) <= delta * V^*(s) and V(s') < V(s) for V^*(s) <= K
        if (options.verbosity>0) std::cout << "Encoding clause C6..." << std::endl;

        for (const auto s:sample_.alive_states()) {
            for (const auto sprime:sample_.successors(s)) {
                if (is_necessarily_bad(get_transition_id(s, sprime))) continue; // includes alive-to-dead transitions
                if (!sample_.is_alive(sprime)) continue;
                if (!sample_.in_sample(sprime)) continue;
                if (get_vstar(s) > K ) continue;

                const auto good_s_prime = variables.goods.at(get_class_representative(s, sprime));

                for (unsigned dprime=1; dprime < max_d; ++dprime) {
                    // (2) Good(s, s') and V(s',dprime) -> V(s) > dprime
                    cnfclause_t clause{Wr::lit(good_s_prime, false),
                                       Wr::lit(v_vars.at({sprime, dprime}), false)};

                    for (unsigned d = dprime + 1; d <= max_d; ++d) {
                        clause.push_back(Wr::lit(v_vars.at({s, d}), true));
                    }
                    wr.cl(clause);
                    ++cl_counter[6];
                }

                // (2') Border condition: V(s', D) implies -Good(s, s')
                wr.cl({Wr::lit(v_vars.at({sprime, max_d}), false), Wr::lit(good_s_prime, false)});
                cl_counter[6]++;
            }
        }

        // C7 Last k-steps must be optimal
        if(options.verbosity>0) std::cout << "Encoding clause C7..." << std::endl;

        std::set<unsigned> srepr;
        for (const auto s : sample_.alive_states()) {
            const auto min_vs = get_vstar(s);
            if (min_vs <= options.optimal_steps ) {
                for (unsigned s_prime:sample_.successors(s)) {
                    auto tx = get_transition_id(s, s_prime);
                    auto repr = get_representative_id(tx);

                    // Avoid non-optimal paths
                    if (get_vstar(s_prime) >= min_vs) {
                        srepr.insert(repr);
                    }
                }
            }
        }
        for (auto repr : srepr) {
            wr.cl({
                          wr.lit(variables.goods.at(repr), false)
                  });
            cl_counter[7]++;
        }

        // C8. Selected(f) optimization
        if(options.verbosity>0) std::cout << "Posting (weighted) soft constraints for " << variables.selecteds.size() << " features" << std::endl;
        for (unsigned f:feature_ids) {
            wr.cl({Wr::lit(variables.selecteds[f], false)}, sample_.feature_weight(f));
        }
        cl_counter[1]+= feature_ids.size();

        // Print a breakdown of the clauses
        std::cout << "A total of " << wr.nclauses() << " clauses were created" << std::endl;
        std::cout << "\t(Weighted) Select(f) [1]: " << cl_counter[1] << std::endl;
        std::cout << "\tD1 and D2 distinguishing features [2]: " << cl_counter[2] << std::endl;
        std::cout << "\tPolicy completeness [3]: " << cl_counter[3] << std::endl;
        std::cout << "\tPolicy is terminating (preliminaries) [4]: " << cl_counter[4] << std::endl;
        std::cout << "\tPolicy is terminating [5]: " << cl_counter[5] << std::endl;
        std::cout << "\tV consistency for V^*(s)<=K [6]: " << cl_counter[6] << std::endl;
        std::cout << "\tOptimality clauses [7]: " << cl_counter[7] << std::endl;

        assert(wr.nclauses() == cl_counter[1] + cl_counter[2] + cl_counter[3] + cl_counter[4] + cl_counter[5] + cl_counter[6] + cl_counter[7] );

        return {CNFGenerationOutput::Success, variables};
    }


} // namespaces
% The overall idea of the encoding is to explicitly search for a finite policy with K rules
% that make use of M features. The primary variables of the encoding represent what is the precondition
% (>0, =0, ?) and effect (INC, DEC, NIL, ?) that each rule J imposes to each selected feature I in order
% for the rule to be compatible with a given transition.
% From these explicitly-represented structure of the rules, we can derive atoms good(S,S') that represent
% whether transition (S,S') in the training set is labeled as good by at least one rule of the policy.
% Based on these atoms good(S,S'), we then rule out any policy that
%    (1) is not defined on all alive states, or
%    (2) is loopy, i.e. has some cycle of good edges.


%% EDB Predicates
%% These will be generated at runtime from information of the sample and feature pool.
%    - alive(X).
%    - unsolvable(X).
%    - transition(X, Y).
%    - feature(F).
%    - weight(F).
%    - pre_f(F,S,X): The value of feature F in state S is X in {gt0, eq0}
%    - eff_f(F,S,S',Y): The value of feature F in transition (S,S') is Y in {inc, dec, nil}

% Output IDB Predicates:
%    - sel(F,I): Feature F selected at "position I" of the abstraction.
%    - pre(J,I,V): The precondition that rule J imposes on selected feature I is V in {gt0, eq0, dontcare}
%    - eff(J,I,V): The effect that rule J imposes on selected feature I is V in {inc, dec, nil, dontcare}


% Num of features:
position(0..m-1).

% Num of rules:
rule(0..k-1).


% Possible precondition and effect values for each rule and position
pre_value(gt0).
pre_value(eq0).
pre_value(dontcare).

eff_value(inc).
eff_value(dec).
eff_value(nil).
eff_value(dontcare).


% PRIMARY "CHOICE" ATOMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

{ sel(F, I) : feature(F) } = 1 :- position(I).
% Note: The constraint below could be dropped, and the effect would be that we
%        can compute policies that use the same feature on different positions,
%        i.e. equivalent to allowing less than M features
{ sel(F, I) : position(I) } <= 1 :- feature(F).


{pre(J, I, V) : pre_value(V)} = 1 :- position(I), rule(J).
{eff(J, I, V) : eff_value(V)} = 1 :- position(I), rule(J).


% DERIVED ATOMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Project the booleanness of each position
bool_i(I) :-  sel(F,I), bool(F).
% num_i(I) :-  sel(F,I), num(F). % Not necessary at the moment

% A ground transition is labeled as good by rule J on position I if the definition of the rule
% for that position is compatible with the actual changes in the ground transition
fulfills_pre(J,I,S) :- alive(S), pre(J,I,dontcare).
fulfills_pre(J,I,S) :- alive(S), sel(F, I), pre(J,I,X), pre_f(F,S,X).

fulfills_eff(J,I,S,S') :- alive(S), transition(S,S'), eff(J,I,dontcare).
fulfills_eff(J,I,S,S') :- alive(S), transition(S,S'), sel(F, I), eff(J,I,X), eff_f(F,S,S',X).

good(J,I,S,S') :- fulfills_pre(J,I,S), fulfills_eff(J,I,S,S').


% Now, we can define good(S,S') as follows: (S, S') is good if for some rule J,
% the values of J in all positions I are consistent with the transition.
% The ASP rule will look as follows, but this will be printed in the instance file, since I didn't find
% a compact way to express this without introducing extra atoms, which at the moment I prefer to avoid.
%%%  good(S,S') :- transition(S,S'), good(J,0,S,S'), good(J,1,S,S'), ...


% Policy completeness: policy is defined in all alive states
:- good(S,S'), unsolvable(S').
defined(S) :- good(S,S'), transition(S,S').
:- not defined(S), alive(S).


% Good edges form no cycle. This makes use of the built-in acyclicity #edge Clingo directive
#edge (S, S') : good(S,S').


% Some optimizations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- transition(S,S), good(S,S).       % No self-loops allowed
:- pre(J,I,eq0), eff(J,I,dec).       % if F=0, then F dec is not possible
% For boolean features P, If P is a precondition, then P is not a possible effect:
:- bool(I), pre(J,I,gt0), eff(J,I,inc).

% Symmetry-breaking constraints to enforce that features are selected in increasing ID.
% Often do more harm than good, since the grounding explodes.
% Comment / uncomment the following line at will.
%:- position(I), position(I+1), sel(F,I), sel(F',I+1), F'<=F.

#show sel/2.
#show pre/3.
#show eff/3.

%#show good/2.

#minimize {W, F: sel(F,I), weight(F, W), position(I)}.

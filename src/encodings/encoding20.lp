%%% Domain encoding
% EDB Predicates:
%%    - alive(X).
%%    - unsolvable(X).
%%    - transition(X, Y).
%%    - feature(F).
%%    - weight(F).
%    - pre_f(F,S,X).
%    - eff_f(F,S,S',Y).

% Output IDB Predicates:
%    - sel(F,I).
%    - pre(J,I,V).
%    - eff(J,I,V).


% Num of features:
#const m = 3.
position(0..m-1).

% Num of rules:
#const k = 3.
rule(0..k-1).


% Possible precondition and effect values for each rule and position
pre_value(gt0).
pre_value(eq0).
pre_value(dontcare).

eff_value(inc).
eff_value(dec).
eff_value(same).
eff_value(dontcare).


% Choice variables: selecteds
%%%%%%%%%%%%%%%%%%

{ sel(F, I) : feature(F) } = 1 :- position(I).
% Note: The constraint below could be dropped, and the effect would be that we
%        can compute abstractions using less than m bits too by selecting the
%        same feature in different positions of the abstraction mapping
{ sel(F, I) : position(I) } = 1 :- feature(F).


% Choice variables: pre and effs of each rule
%%%%%%%%%%%%%%%%%%
{pre(J, I, V) : pre_value(V)} = 1 :- position(I), rule(J).
{eff(J, I, V) : eff_value(V)} = 1 :- position(I), rule(J).


% Derived variables: Good(i, j, s, s')  and Good(s, s')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A ground transition is labeled as good by rule J on position I if the definition of the rule
% for that position is compatible with the actual changes in the ground transition
good(J,I,S,S') :- alive(S), transition(S,S'), sel(F, I), pre(J,I,X), pre_f(F,S,X), eff(J,I,Y), eff_f(F,S,S',Y).


% A ground transition is Good if for some rule J, the values of J in all positions I are consistent with the transition.
good(S,S') :- good(J,I,S,S') : position(I) : rule(J).


% Policy completeness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- good(S,S'), unsolvable(S').
defined(S) :- good(S,S'), transition(S,S').
:- defined(S), alive(S).


% Good edges form no cycle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#edge (S, S') : good(S,S').


#show sel/2.
#show good/2.
#show pre/3.
#show eff/3.

#minimize {W, F: sel(F), weight(F, W)}.

%%% Domain encoding
% EDB Predicates:
%    - alive(X).
%    - unsolvable(X).
%    - transition(X, Y).
%    - feature(F).
%    - weight(F).
%    - pre_f(F,S,X).
%    - eff_f(F,S,S',Y).

% Output IDB Predicates:
%    - sel(F,I).
%    - pre(J,I,V).
%    - eff(J,I,V).


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


% Choice variables: selecteds
%%%%%%%%%%%%%%%%%%

{ sel(F, I) : feature(F) } = 1 :- position(I).
% Note: The constraint below could be dropped, and the effect would be that we
%        can compute policies that use the same feature on different positions,
%        i.e. equivalent to allowing less than M features
% { sel(F, I) : position(I) } <= 1 :- feature(F).


% Choice variables: pre and effs of each rule
%%%%%%%%%%%%%%%%%%
{pre(J, I, V) : pre_value(V)} = 1 :- position(I), rule(J).
{eff(J, I, V) : eff_value(V)} = 1 :- position(I), rule(J).

% Project feature value into the actual feature selection
val_i(I,S,X) :- sel(F,I), val(F,S,X).
bool_i(I) :-  sel(F,I), bool(F).
num_i(I) :-  sel(F,I), num(F).

% Derived variables: Good(i, j, s, s')  and Good(s, s')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A ground transition is labeled as good by rule J on position I if the definition of the rule
% for that position is compatible with the actual changes in the ground transition
fulfills_pre(J,I,S) :- alive(S), pre(J,I,dontcare).
fulfills_pre(J,I,S) :- alive(S), pre(J,I,gt0), val_i(I,S,X), X>0.
fulfills_pre(J,I,S) :- alive(S), pre(J,I,eq0), val_i(I,S,0).

fulfills_eff(J,I,S,S') :- alive(S), transition(S,S'), eff(J,I,dontcare).
fulfills_eff(J,I,S,S') :- alive(S), transition(S,S'), eff(J,I,inc), val_i(I,S,X), val_i(I,S',Y), X<Y.
fulfills_eff(J,I,S,S') :- alive(S), transition(S,S'), eff(J,I,nil), val_i(I,S,X), val_i(I,S',X).
fulfills_eff(J,I,S,S') :- alive(S), transition(S,S'), eff(J,I,dec), val_i(I,S,X), val_i(I,S',Y), X>Y.

good(J,I,S,S') :- fulfills_pre(J,I,S), fulfills_eff(J,I,S,S').


% A ground transition is Good if for some rule J, the values of J in all positions I are consistent with the transition.
% NOTE: This will be printed in the instance file, one ASP rule per policy rule J:
%%%  good(S,S') :- transition(S,S'), good(J,0,S,S'), good(J,1,S,S'), ...


% Policy completeness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- good(S,S'), unsolvable(S').
defined(S) :- good(S,S'), transition(S,S').
:- not defined(S), alive(S).


% Good edges form no cycle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#edge (S, S') : good(S,S').


% Some optimizations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- transition(S,S), good(S,S).       % No self-loops allowed
:- pre(J,I,eq0), eff(J,I,dec).       % if F=0, then F dec is not possible

:- bool(I), pre(J,I,gt0), eff(J,I,inc).

% Symmetry-breaking constraints. Often do more harm than good. ¿?
%:- position(I), position(I+1), sel(F,I), sel(F',I+1), F'<=F.

#show sel/2.
#show pre/3.
#show eff/3.

%#show good/2.

#minimize {W, F: sel(F,I), weight(F, W), position(I)}.

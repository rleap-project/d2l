
%%% Domain encoding

% Auxiliary definitions
% d1: state distinguishability
d1(S, T) :- selected(F), state(S), state(T), val(F, S, 0), not val(F, T, 0), S<T.
d1(S, T) :- selected(F), state(S), state(T), val(F, T, 0), not val(F, S, 0), S<T.

% d2: transition distinguishability
change(F, S, S', 1)  :- transition(S, S'), val(F, S, V), val(F, S', V'), V < V'.
change(F, S, S', -1) :- transition(S, S'), val(F, S, V), val(F, S', V'), V > V'.
change(F, S, S', 0)  :- transition(S, S'), val(F, S, V), val(F, S', V'), V = V'.

d2(S, S', T, T') :- selected(F), change(F, S, S', X), change(F, T, T', Y), X!=Y.
d2(S, S', T, T') :- d1(S, T), transition(S, S'), transition(T, T').
d2(S, S', T, T') :- d1(T, S), transition(S, S'), transition(T, T').


% Variable Definitions
% C1. The policy has to be complete with respect to all alive states
1 { good(S,S') : transition(S,S') } :- alive(S).

{ selected(F) } :- feature(F).

v(S,0) :- goal(S).
{ v(S,0..maxd) } = 1 :- alive(S).


% C3-4 V values are constrained by Vstar
:- alive(S), v(S, D), vstar(S, Dstar), D < Dstar.
:- alive(S), v(S, D), vstar(S, Dstar), D > delta*Dstar.


% C2. V is always descending along Good transitions:
:- good(S,S'), v(S,VS), v(S', VS'), VS' >= VS.


% C7 Goals are distinguishable from non-goals.
:- goal(S), state(T), not goal(T), not d1(S, T), not d1(T, S).


% C5-6. Good transitions can be distinguished from bad transitions.
:- transition(T, T'), good(S, S'), not good(T, T'), not d2(S, S', T, T'), not d2(T, T', S, S').


#show selected/1.
% #show good/2.

#minimize {W, F: selected(F), weight(F, W)}.

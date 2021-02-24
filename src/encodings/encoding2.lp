
%%% Domain encoding

#const m = 3.

position(0..m-1).
sigma(0..2**m-1).

% C1
%%%%%%%%%%%%%%%%%%
{ selected(F, I) : feature(F) } = 1 :- position(I).

% Note: The constraint below could be dropped, and the effect would be that we
%        can compute abstractions using less than m bits too by selecting the
%        same feature in different positions of the abstraction mapping
{ selected(F, I) : position(I) } = 1 :- feature(F).

selected(F) :- selected(F, I).

% C2
%%%%%%%%%%%%%%%%%%
% We want maps to be consistent with the feature selection.
{ maps(S, Sigma) : sigma(Sigma)} = 1 :- state(S).
:- maps(S, Sigma), selected(F, I), val(F, S, V), V>0, Sigma & (2**I) = 0.
:- maps(S, Sigma), selected(F, I), val(F, S, V), V=0, Sigma & (2**I) > 0.

% C3 Goal and unsolvable-transition distinguishability
%%%%%%%%%%%%%%%%%%
:- goal(S), state(T), not goal(T), not d1(S, T), not d1(T, S).
:- good(S, S'), transition(T, T'), alive(T), dead(T'), not d2(S, S', T, T'), not d2(T, T', S, S').
:- good(S, S'), alive(S), dead(S').

% C4
%%%%%%%%%%%%%%%%%%
1 { good(S,S') : transition(S,S') } :- alive(S).

% C6
%%%%%%%%%%%%%%%%%%
% C6.a At least one feature changes in every good transition
change(S, S') :- transition(S, S'), selected(F), delta(F, S, S', X), X!=0.
:- not change(S, S'), good(S, S').


% Termination
%%%%%%%%%%%%%%%%%%
% Preliminaries
delta(I, S, X) :- good(S, S'), selected(F, I), delta(F, S, S', X) .
incs(I, Sigma) :- delta(I, S, 1), maps(S, Sigma).
decs(I, Sigma) :- delta(I, S, -1), maps(S, Sigma).
mods(I, Sigma) :- incs(I, Sigma).
mods(I, Sigma) :- decs(I, Sigma).

incs(I) :- incs(I, Sigma).
decs(I) :- decs(I, Sigma).

% Base case: a feature position that never increases or never decreases is split
split(I) :- not incs(I), position(I).
split(I) :- not decs(I), position(I).

% Induction:
% If feature at J is split and has diff values in sigma and sigma', the pair
% of abstract states is "saved" from looping.
saved(Sigma, Sigma') :- split(J), Sigma & (2**J) != Sigma' & (2**J), Sigma < Sigma', position(Sigma), position(Sigma').

% If feature at J is split and sigma or sigma' modify its value, the pair
% of abstract states is "saved" from looping.
saved(Sigma, Sigma') :- split(J), mods(Sigma, J), position(Sigma'), Sigma < Sigma'.
saved(Sigma, Sigma') :- split(J), mods(Sigma', J), position(Sigma), Sigma < Sigma'.

% If feature #I increases at some Sigma, decreases at another, and the pair is
% not saved, the I cannot be split
-split(I) :- incs(Sigma, I), decs(Sigma', I), not saved(Sigma, Sigma').

% Auxiliary definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% d1: state distinguishability
d1(S, T) :- selected(F), state(S), state(T), val(F, S, 0), not val(F, T, 0), S<T.
d1(S, T) :- selected(F), state(S), state(T), val(F, T, 0), not val(F, S, 0), S<T.

% d2: transition distinguishability
d2(S, S', T, T') :- selected(F), delta(F, S, S', X), delta(F, T, T', Y), X!=Y.
d2(S, S', T, T') :- d1(S, T), transition(S, S'), transition(T, T').
d2(S, S', T, T') :- d1(T, S), transition(S, S'), transition(T, T').

% delta: what is the change of feature F in transition S, S'
delta(F, S, S', 1)  :- transition(S, S'), val(F, S, V), val(F, S', V'), V < V'.
delta(F, S, S', -1) :- transition(S, S'), val(F, S, V), val(F, S', V'), V > V'.
delta(F, S, S', 0)  :- transition(S, S'), val(F, S, V), val(F, S', V'), V = V'.


%#show selected/1.
%#show selected/2.
% #show good/2.

#minimize {W, F: selected(F), weight(F, W)}.


%%% Domain encoding

% EDB Predicates:
%    - sampled(X).
%    - alive(X).
%    - dead(X).
%    - transition(X, Y).
%    - feature(F).

% Output IDB Predicates:
%    - sel(F).
%    - good(X, Y).


% Choose at least one outgoing Good transition on each alive state
1 { good(T, T') : transition(S,S'), repr(S, S', T, T') } :- alive(S), sampled(S).

% Choose whether to select feature f or not
{ sel(F) } :- feature(F).

% Good edges form no cycle
#edge (S, S') : good(S,S').

% DEBUGGING
sa(S) :- alive(S), sampled(S).
out(S, S') :- sa(S), transition(S, S').
todist(X, X',Y, Y') :- good(S, S'), not good(T, T'), transition(T, T'), repr(S, S', X, X'), repr(T, T', Y, Y').

#show sel/1.
#show good/2.
%#show sa/1.
%#show out/2.
%#show todist/4.


#minimize {W, F: sel(F), weight(F, W)}.

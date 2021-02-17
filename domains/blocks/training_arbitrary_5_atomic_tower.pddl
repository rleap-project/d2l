(define (problem BLOCKS-5-0)
(:domain blocksworld-atomic)
(:objects a b c d e)

(:init
    (on a table)
    (on b table)
    (on c table)
    (on d table)
    (on e table)
    (clear a)
    (clear b)
    (clear c)
    (clear d)
    (clear e)
    (clear table)

;; import itertools
;; _ = [print(f"(diff {b1} {b2})") for b1, b2 in itertools.permutations(['table'] + 'a b c d e'.split(), 2)]
    (diff table a)
    (diff table b)
    (diff table c)
    (diff table d)
    (diff table e)
    (diff a table)
    (diff a b)
    (diff a c)
    (diff a d)
    (diff a e)
    (diff b table)
    (diff b a)
    (diff b c)
    (diff b d)
    (diff b e)
    (diff c table)
    (diff c a)
    (diff c b)
    (diff c d)
    (diff c e)
    (diff d table)
    (diff d a)
    (diff d b)
    (diff d c)
    (diff d e)
    (diff e table)
    (diff e a)
    (diff e b)
    (diff e c)
    (diff e d)
)

(:goal (and
    (on a b)
	(on b c)
	(on c d)
	(on d e)
	(on e table)
))






)
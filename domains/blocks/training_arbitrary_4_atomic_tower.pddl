(define (problem BLOCKS-4-0)
(:domain blocksworld-atomic)
(:objects a b c d )

(:init
    (on a table)
    (on b table)
    (on c table)
    (on d table)
    (clear a)
    (clear b)
    (clear c)
    (clear d)
    (clear table)

;; import itertools
;; _ = [print(f"(diff {b1} {b2})") for b1, b2 in itertools.permutations(['table'] + 'a b c d'.split(), 2)]
    (diff table a)
    (diff table b)
    (diff table c)
    (diff table d)
    (diff a table)
    (diff a b)
    (diff a c)
    (diff a d)
    (diff b table)
    (diff b a)
    (diff b c)
    (diff b d)
    (diff c table)
    (diff c a)
    (diff c b)
    (diff c d)
    (diff d table)
    (diff d a)
    (diff d b)
    (diff d c)
)

(:goal (and
    (on a b)
	(on b c)
	(on c d)
	(on d table)
))






)

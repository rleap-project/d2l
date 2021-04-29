
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem instance-6-0)
    (:domain blocksworld-atomic)

    (:objects
        b1 b2 b3 b4 b5 b6 - object
    )

    (:init
        (on b5 table)
        (on b4 table)
        (on b1 b5)
        (on b3 b4)
        (on b2 table)
        (on b6 b1)
        (clear b3)
        (clear table)
        (clear b6)
        (clear b2)
        (diff table b5)
        (diff b3 b1)
        (diff b4 b2)
        (diff b2 b5)
        (diff b2 b6)
        (diff table b2)
        (diff b5 b1)
        (diff b6 table)
        (diff b3 table)
        (diff b4 b1)
        (diff b5 table)
        (diff b1 table)
        (diff b6 b3)
        (diff table b1)
        (diff b4 table)
        (diff b2 b1)
        (diff b6 b4)
        (diff b5 b3)
        (diff b3 b4)
        (diff b1 b3)
        (diff b2 table)
        (diff b5 b4)
        (diff b1 b4)
        (diff b4 b3)
        (diff table b3)
        (diff b2 b3)
        (diff b3 b6)
        (diff b6 b5)
        (diff b3 b5)
        (diff table b4)
        (diff b2 b4)
        (diff b5 b6)
        (diff b1 b6)
        (diff b6 b2)
        (diff b3 b2)
        (diff b1 b5)
        (diff b4 b5)
        (diff b4 b6)
        (diff b5 b2)
        (diff b1 b2)
        (diff table b6)
        (diff b6 b1)
    )

    (:goal
        (and (on b5 table) (on b2 b5) (on b6 b2) (on b4 table) (on b1 b4) (on b3 b1))
    )

    
    
    
)


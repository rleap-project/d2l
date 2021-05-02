
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem instance-8-0)
    (:domain blocksworld-atomic)

    (:objects
        b1 b2 b3 b4 b5 b6 b7 b8 - object
    )

    (:init
        (on b5 table)
        (on b8 b3)
        (on b1 b6)
        (on b6 b7)
        (on b7 b2)
        (on b4 b5)
        (on b3 table)
        (on b2 table)
        (clear b8)
        (clear b4)
        (clear table)
        (clear b1)
        (diff table b5)
        (diff table b8)
        (diff b3 b1)
        (diff b4 b2)
        (diff b2 b5)
        (diff b2 b6)
        (diff b6 b7)
        (diff b7 b3)
        (diff table b2)
        (diff b2 b8)
        (diff b5 b1)
        (diff b3 b7)
        (diff b8 b6)
        (diff b6 table)
        (diff b3 table)
        (diff b8 b5)
        (diff b4 b1)
        (diff b5 b7)
        (diff b1 b7)
        (diff b7 b4)
        (diff b5 table)
        (diff b4 b7)
        (diff b1 table)
        (diff b8 b2)
        (diff b6 b3)
        (diff table b1)
        (diff b4 table)
        (diff b2 b1)
        (diff table b7)
        (diff b6 b4)
        (diff b5 b3)
        (diff b3 b4)
        (diff b1 b3)
        (diff b2 b7)
        (diff b7 b6)
        (diff b8 b1)
        (diff b2 table)
        (diff b7 b5)
        (diff b7 b8)
        (diff b5 b4)
        (diff b1 b4)
        (diff b4 b3)
        (diff b8 b7)
        (diff b7 b2)
        (diff table b3)
        (diff b8 table)
        (diff b2 b3)
        (diff b3 b6)
        (diff b6 b5)
        (diff b6 b8)
        (diff b3 b5)
        (diff b3 b8)
        (diff table b4)
        (diff b2 b4)
        (diff b5 b6)
        (diff b1 b6)
        (diff b6 b2)
        (diff b8 b3)
        (diff b3 b2)
        (diff b7 b1)
        (diff b5 b8)
        (diff b1 b5)
        (diff b1 b8)
        (diff b4 b5)
        (diff b4 b6)
        (diff b8 b4)
        (diff b5 b2)
        (diff b1 b2)
        (diff b7 table)
        (diff b4 b8)
        (diff table b6)
        (diff b6 b1)
    )

    (:goal
        (and (on b8 table) (on b2 table) (on b7 b8) (on b6 table) (on b1 b6) (on b5 b2) (on b4 table) (on b3 b5))
    )

    
    
    
)

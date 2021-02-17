(define (problem BLOCKS-11-1)
(:domain BLOCKS)
(:objects B C E A H K I G D F J )
(:INIT (CLEAR J) (CLEAR F) (CLEAR D) (CLEAR G) (ONTABLE I) (ONTABLE K)
 (ONTABLE H) (ONTABLE A) (ON J I) (ON F E) (ON E K) (ON D C) (ON C H) (ON G B)
 (ON B A) (HANDEMPTY))
(:goal (AND (CLEAR A)))

)
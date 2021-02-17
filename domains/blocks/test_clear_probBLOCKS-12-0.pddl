(define (problem BLOCKS-12-0)
(:domain BLOCKS)
(:objects I D B E K G A F C J L H )
(:INIT (CLEAR H) (CLEAR L) (CLEAR J) (ONTABLE C) (ONTABLE F) (ONTABLE J)
 (ON H A) (ON A G) (ON G K) (ON K E) (ON E B) (ON B D) (ON D I) (ON I C)
 (ON L F) (HANDEMPTY))
(:goal (AND (CLEAR A)))

)
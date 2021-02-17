(define (problem BLOCKS-9-1)
(:domain BLOCKS)
(:objects H G I C D B E A F )
(:INIT (CLEAR F) (ONTABLE A) (ON F E) (ON E B) (ON B D) (ON D C) (ON C I)
 (ON I G) (ON G H) (ON H A) (HANDEMPTY))
(:goal (AND (CLEAR A)))

)
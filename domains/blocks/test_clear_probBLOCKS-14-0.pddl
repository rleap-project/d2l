(define (problem BLOCKS-14-0)
(:domain BLOCKS)
(:objects I D B L C K M H J N E F G A )
(:INIT (CLEAR A) (CLEAR G) (CLEAR F) (ONTABLE E) (ONTABLE N) (ONTABLE F)
 (ON A J) (ON J H) (ON H M) (ON M K) (ON K C) (ON C L) (ON L B) (ON B E)
 (ON G D) (ON D I) (ON I N) (HANDEMPTY))
(:goal (AND (CLEAR A)))

)
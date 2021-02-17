(define (problem BLOCKS-11-0)
(:domain BLOCKS)
(:objects F A K H G E D I C J B )
(:INIT (CLEAR B) (CLEAR J) (CLEAR C) (ONTABLE I) (ONTABLE D) (ONTABLE E)
 (ON B G) (ON G H) (ON H K) (ON K A) (ON A F) (ON F I) (ON J D) (ON C E)
 (HANDEMPTY))
(:goal (AND (ON A B)))

)
(define (problem BLOCKS-10-1)
(:domain BLOCKS)
(:objects D A J I E G H B F C )
(:INIT (CLEAR C) (CLEAR F) (ONTABLE B) (ONTABLE H) (ON C G) (ON G E) (ON E I)
 (ON I J) (ON J A) (ON A B) (ON F D) (ON D H) (HANDEMPTY))
(:goal (AND (ON A B)))

)
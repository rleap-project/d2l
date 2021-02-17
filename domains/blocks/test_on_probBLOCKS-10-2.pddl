(define (problem BLOCKS-10-2)
(:domain BLOCKS)
(:objects B G E D F H I A C J )
(:INIT (CLEAR J) (CLEAR C) (ONTABLE A) (ONTABLE C) (ON J I) (ON I H) (ON H F)
 (ON F D) (ON D E) (ON E G) (ON G B) (ON B A) (HANDEMPTY))
(:goal (AND (ON A B)))

)
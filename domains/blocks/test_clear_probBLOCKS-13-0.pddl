(define (problem BLOCKS-13-0)
(:domain BLOCKS)
(:objects L H E A J C D F G K M I B )
(:INIT (CLEAR B) (CLEAR I) (CLEAR M) (ONTABLE K) (ONTABLE G) (ONTABLE M)
 (ON B F) (ON F D) (ON D C) (ON C J) (ON J A) (ON A E) (ON E H) (ON H L)
 (ON L K) (ON I G) (HANDEMPTY))
(:goal (AND (CLEAR A)))

)
(define (problem BLOCKS-8-2)
(:domain BLOCKS)
(:objects F B G C H E A D )
(:INIT (CLEAR D) (CLEAR A) (CLEAR E) (CLEAR H) (CLEAR C) (ONTABLE G)
 (ONTABLE A) (ONTABLE E) (ONTABLE H) (ONTABLE C) (ON D B) (ON B F) (ON F G)
 (HANDEMPTY))
(:goal (AND (CLEAR A)))
)
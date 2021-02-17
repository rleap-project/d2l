(define (problem BLOCKS-12-1)
(:domain BLOCKS)
(:objects E L A B F I H G D J K C )
(:INIT (CLEAR C) (CLEAR K) (ONTABLE J) (ONTABLE D) (ON C G) (ON G H) (ON H I)
 (ON I F) (ON F B) (ON B A) (ON A L) (ON L E) (ON E J) (ON K D) (HANDEMPTY))
(:goal (AND (CLEAR A)))

)
(define (problem maintenance-scheduling-1-3-3-4-2)
 (:domain maintenance-scheduling-domain)
 (:objects d1 d2 d3 d4 - day
   FRA BER HAM - airport
   ap1 ap2 ap3 ap4 - plane
   v1 v2 v3 v4 v5 v6 v7 v8 - visit
   )
 (:init
  (today d1)  (today d2)  (today d3) 
  ;;(at ap1 d1 BER)
  (which v1 ap1) (where v1 BER) (when_ v1 d1)
  
  ;;(at ap1 d2 BER)
  (which v2 ap1) (where v2 BER) (when_ v2 d2)
  
  ;;(at ap2 d2 FRA)
  (which v3 ap2) (where v3 FRA) (when_ v3 d2)
  
  ;;(at ap2 d3 BER)
  (which v4 ap2) (where v4 BER) (when_ v4 d3)
  
  ;;(at ap3 d1 BER)
  (which v5 ap3) (where v5 BER) (when_ v5 d1)
  
  ;;(at ap3 d3 BER)
  (which v6 ap3) (where v6 BER) (when_ v6 d3)
  
  ;;(at ap4 d3 BER)
  (which v7 ap4) (where v7 BER) (when_ v7 d3)
  
  ;;(at ap4 d3 BER)
  (which v8 ap4) (where v8 BER) (when_ v8 d3)
)
  (:goal (forall (?plane - plane) (done ?plane)))
)

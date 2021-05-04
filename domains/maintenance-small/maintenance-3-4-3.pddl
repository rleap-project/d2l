(define (problem maintenance-scheduling-1-3-3-4-2)
 (:domain maintenance-scheduling-domain)
 (:objects d1 d2 d3 d4 - day
   FRA BER HAM - airport
   ap1 ap2 ap3 ap4 - plane)
 (:init
  (today d1)  (today d2)  (today d3)  (at ap1 d1 BER)
  (at ap1 d2 BER)
  (at ap2 d2 FRA)
  (at ap2 d3 BER)
  (at ap3 d1 BER)
  (at ap3 d3 BER)
  (at ap4 d3 BER)
  (at ap4 d3 BER)
)
  (:goal (forall (?plane - plane) (done ?plane)))
  )

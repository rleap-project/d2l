(define (problem maintenance-scheduling-1-3-10-10-2-0)
 (:domain maintenance-scheduling-domain)
 (:objects d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 - day
   FRA BER HAM - airport
   ap1 ap2 ap3 ap4 ap5 ap6 ap7 ap8 ap9 ap10 - plane
   v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 - visit

   )
 (:init
  (today d1)  (today d2)  (today d3)  (today d4)  (today d5)  (today d6)  (today d7)  (today d8)  (today d9)  (today d10)

    ;; (at ap9 d10 fra)
    (which v1 ap9) (where v1 fra) (when_ v1 d10)

    ;; (at ap5 d3 ber)
    (which v2 ap5) (where v2 ber) (when_ v2 d3)

    ;; (at ap2 d7 fra)
    (which v3 ap2) (where v3 fra) (when_ v3 d7)

    ;; (at ap5 d1 fra)
    (which v4 ap5) (where v4 fra) (when_ v4 d1)

    ;; (at ap4 d1 ber)
    (which v5 ap4) (where v5 ber) (when_ v5 d1)

    ;; (at ap6 d8 fra)
    (which v6 ap6) (where v6 fra) (when_ v6 d8)

    ;; (at ap8 d8 ber)
    (which v7 ap8) (where v7 ber) (when_ v7 d8)

    ;; (at ap2 d4 ber)
    (which v8 ap2) (where v8 ber) (when_ v8 d4)

    ;; (at ap1 d4 ber)
    (which v9 ap1) (where v9 ber) (when_ v9 d4)

    ;; (at ap6 d2 ham)
    (which v10 ap6) (where v10 ham) (when_ v10 d2)

    ;; (at ap10 d2 ber)
    (which v11 ap10) (where v11 ber) (when_ v11 d2)

    ;; (at ap1 d8 ber)
    (which v12 ap1) (where v12 ber) (when_ v12 d8)

    ;; (at ap3 d3 ber)
    (which v13 ap3) (where v13 ber) (when_ v13 d3)

    ;; (at ap8 d10 fra)
    (which v14 ap8) (where v14 fra) (when_ v14 d10)

    ;; (at ap9 d3 fra)
    (which v15 ap9) (where v15 fra) (when_ v15 d3)

    ;; (at ap7 d3 ber)
    (which v16 ap7) (where v16 ber) (when_ v16 d3)

    ;; (at ap10 d4 ber)
    (which v17 ap10) (where v17 ber) (when_ v17 d4)

    ;; (at ap7 d3 ham)
    (which v18 ap7) (where v18 ham) (when_ v18 d3)

    ;; (at ap4 d4 ber)
    (which v19 ap4) (where v19 ber) (when_ v19 d4)

    ;; (at ap3 d10 ber)
    (which v20 ap3) (where v20 ber) (when_ v20 d10)
)
  (:goal (and 
 (done ap1)
 (done ap2)
 (done ap3)
 (done ap4)
 (done ap5)
 (done ap6)
 (done ap7)
 (done ap8)
 (done ap9)
 (done ap10)
  )) 
  )

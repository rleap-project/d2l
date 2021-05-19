; There are mechanics who on any day may work at one
; of several cities where the airplane maintenance
; company has facilities. There are airplanes each of
; which has to be maintained during the given time period.
; The airplanes are guaranteed to visit some of the cities
; on given days. The problem is to schedule the presence
; of the mechanics so that each plane will get maintenance.

(define (domain maintenance-scheduling-domain)
 (:requirements :adl :typing :conditional-effects)
 (:types plane day airport visit)
 (:predicates
   (done ?p - plane)
   (today ?d - day)

   (when_  ?v - visit ?d - day)
   (where ?v - visit ?a - airport)
   (which ?v - visit ?p - plane)
  )

 (:action workat
  :parameters (?day - day ?airport - airport)
  :precondition (today ?day)
  :effect (and
     (not (today ?day))
     (forall (?plane - plane)
        (when (exists (?v - visit) (and (when_ ?v ?day) (where ?v ?airport) (which ?v ?plane)))
              (done ?plane)))
  ))

)

(define (domain Navigation)
  (:requirements :strips)
  (:predicates (agent ?p1)
  (adjacent ?p1 ?p2)
  (wall ?p1)
  (reward ?p2)           
  )

  (:action move
         :parameters (?p1 ?p2)
         :precondition (and (not (wall ?p2)) (adjacent ?p1 ?p2) (agent ?p1))
         :effect (and (not (agent ?p1)) (agent ?p2) (not (reward ?p2)) )
  )
 
)
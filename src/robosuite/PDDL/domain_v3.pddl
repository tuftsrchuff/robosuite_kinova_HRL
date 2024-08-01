(define (domain hanoi_parts_3)
(:requirements :strips :typing)
(:types
    taskstatus - boolean
    disc peg - stackable
    gripper
)
(:predicates 
(clear ?disc - stackable)
(on ?discsm - disc ?disclg - stackable) 
(smaller ?disclg - stackable ?discsm - disc) 
(over ?disc - stackable) 
(grasped ?disc - stackable)
(free ?gripper - gripper)
)



(:action reach_pick
:parameters (?disc - disc ?from - stackable ?to - stackable ?gripper - gripper)
:precondition (and 
    (clear ?disc) 
    (clear ?to)
    (smaller ?to ?disc)
    (on ?disc ?from)
    (free ?gripper)
    (forall (?place - stackable) (not (over ?place)))
    )
:effect (and (over ?disc) (over ?from) (not (over ?to)))
)

(:action pick
:parameters (?disc - disc ?from - stackable ?to - stackable ?gripper - gripper)
:precondition (and 
    (clear ?disc)
    (clear ?to)
    (over ?disc) 
    (over ?from)
    (smaller ?to ?disc)
    (on ?disc ?from)
    (free ?gripper)
    )
:effect (and 
    (grasped ?disc) 
    (not (clear ?disc))
    (not (on ?disc ?from))
    (not (free ?gripper))
    (clear ?from)
    )
)

(:action reach_drop
:parameters (?disc - disc ?from - stackable ?to - stackable ?gripper - gripper)
:precondition (and 
    (grasped ?disc) 
    (clear ?to) 
    (smaller ?to ?disc)
    (clear ?from)
    (over ?from)
    )
:effect (and (over ?to) (not (over ?from)))
)

(:action drop
:parameters (?disc - disc ?from - stackable ?to - stackable ?gripper - gripper)
:precondition (and 
    (over ?to) 
    (grasped ?disc) 
    (clear ?to) 
    (smaller ?to ?disc)
    )
:effect (and 
    (not (grasped ?disc)) 
    (not (clear ?to)) 
    (clear ?disc) 
    (on ?disc ?to) 
    (not (over ?disc)) 
    (not (over ?to))
    (free ?gripper)
    (not (over ?from))
    )
)
)
(define (domain hanoi_parts)
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
(inprogress ?task - taskstatus)
(closegripper ?gripper - gripper)
)




(:action reach_pick
:parameters (?disc - disc ?to - stackable ?from - stackable ?task - taskstatus ?gripper - gripper)
:precondition (and 
    (clear ?disc) 
    (not (over ?disc)) 
    (clear ?to) 
    (not (grasped ?to)) 
    (not (grasped ?from)) 
    (not (grasped ?disc)) 
    (smaller ?to ?disc)
    (on ?disc ?from)
    (not (inprogress ?task))
    (not (closegripper ?gripper))
    )
:effect (and (over ?disc) (inprogress ?task))
)

(:action pick
:parameters (?disc - disc ?to - stackable ?from - stackable ?task - taskstatus ?gripper - gripper)
:precondition (and 
    (clear ?disc) 
    (clear ?to) 
    (not (grasped ?to)) 
    (not (grasped ?from)) 
    (not (grasped ?disc)) 
    (over ?disc) 
    (smaller ?to ?disc)
    (on ?disc ?from)
    (inprogress ?task)
    (not (closegripper ?gripper))
    )
:effect (and 
    (grasped ?disc) 
    (clear ?from) 
    (not (clear ?disc))
    (not (on ?disc ?from))
    (closegripper ?gripper)
    )
)

(:action reach_drop
:parameters (?disc - disc ?to - stackable ?from - stackable ?task - taskstatus ?gripper - gripper)
:precondition (and 
    (grasped ?disc) 
    (clear ?to) 
    (smaller ?to ?disc)
    (inprogress ?task)
    )
:effect (and 
    (over ?to) 
    (not (over ?from))
    )
)

(:action drop
:parameters (?disc - disc ?to - stackable ?from - stackable ?task - taskstatus ?gripper - gripper)
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
    (not (inprogress ?task))
    (not (closegripper ?gripper))
    )
)
)
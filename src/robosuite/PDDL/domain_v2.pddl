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
(reach_pick ?disc - disc)
(pick ?disc - disc)
(reach_drop ?disc)
)




(:action reach_pick
:parameters (?disc - disc ?from - stackable ?task - taskstatus ?gripper - gripper)
:precondition (and 
    (clear ?disc) 
    (not (over ?disc)) 
    (not (grasped ?from)) 
    (not (grasped ?disc)) 
    (on ?disc ?from)
    (not (inprogress ?task))
    (not (closegripper ?gripper))
    )
:effect (and (over ?disc) (inprogress ?task) (reach_pick ?disc))
)

(:action pick
:parameters (?disc - disc ?from - stackable ?task - taskstatus ?gripper - gripper)
:precondition (and 
    (clear ?disc)
    (not (grasped ?from)) 
    (not (grasped ?disc)) 
    (over ?disc) 
    (on ?disc ?from)
    (inprogress ?task)
    (not (closegripper ?gripper))
    (reach_pick ?disc)
    )
:effect (and 
    (grasped ?disc) 
    (clear ?from) 
    (not (clear ?disc))
    (not (on ?disc ?from))
    (closegripper ?gripper)
    (not (reach_pick ?disc))
    (pick ?disc)
    )
)

(:action reach_drop
:parameters (?disc - disc ?to - stackable ?task - taskstatus ?gripper - gripper)
:precondition (and 
    (grasped ?disc) 
    (clear ?to) 
    (smaller ?to ?disc)
    (inprogress ?task)
    (pick ?disc)
    )
:effect (and 
    (over ?to) 
    (not (pick ?disc))
    (reach_drop ?disc)
    )
)

(:action drop
:parameters (?disc - disc ?to - stackable ?task - taskstatus ?gripper - gripper)
:precondition (and 
    (over ?to) 
    (grasped ?disc) 
    (clear ?to) 
    (smaller ?to ?disc)
    (reach_drop ?disc)
    (clear ?to)
    (reach_drop ?disc)
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
    (not (reach_drop ?disc))
    )
)
)
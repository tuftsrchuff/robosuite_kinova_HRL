(define (problem hanoi)
(:domain hanoi_parts)
(:objects 
    peg1 - stackable
    peg2 - stackable
    peg3 - stackable
    cube1 - disc
    cube2 - disc
    cube3 - disc
    task - taskstatus
    gripper - gripper
    )
(:init
(smaller peg1 cube1) (smaller peg1 cube2) (smaller peg1 cube3)
(smaller peg2 cube1) (smaller peg2 cube2) (smaller peg2 cube3)
(smaller peg3 cube1) (smaller peg3 cube2) (smaller peg3 cube3)
(smaller cube2 cube1) (smaller cube3 cube1) (smaller cube3 cube2)
(clear peg2) (clear peg3) (clear cube1)
(on cube3 peg1) (on cube2 cube3) (on cube1 cube2)
)
(:goal (and (on cube3 peg3) (on cube2 cube3) (on cube1 cube2)))
)
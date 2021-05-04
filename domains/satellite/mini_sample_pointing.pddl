(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0	
	satellite1
	instrument0
	instrument1
	thermograph0
	thermograph1
	GroundStation1
	GroundStation2
	GroundStation3
)
(:init
	(satellite satellite0)
	(instrument instrument0)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 GroundStation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation2)
	
	(satellite satellite1)
	(instrument instrument1)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 GroundStation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation2)
		
	(mode thermograph0)
	(mode thermograph1)
	
	(direction GroundStation2)
	(direction GroundStation1)
	(direction GroundStation3)
)
(:goal (and
	(pointing satellite0 GroundStation1)
	(pointing satellite1 GroundStation3)
))

)

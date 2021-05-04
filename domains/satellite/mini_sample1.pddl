(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0	
	satellite1
	instrument0
	instrument1
	thermograph0
	thermograph1
	GroundStation2
	Phenomenon4
	Star5
)
(:init
	(satellite satellite0)
	(instrument instrument0)
	(instrument instrument1)
	(supports instrument0 thermograph0)
	(supports instrument1 thermograph1)	
	(calibration_target instrument0 GroundStation2)
	(calibration_target instrument1 GroundStation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation2)
		
	(mode thermograph0)
	(mode thermograph1)
	
	(direction GroundStation2)
	(direction Phenomenon4)
	(direction Star5)
	
	(satellite satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation2)
)
(:goal (and
	(have_image Phenomenon4 thermograph0)
	(have_image Star5 thermograph0)
	(have_image Star5 thermograph1)
))

)

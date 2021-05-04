(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite4	
	satellite5
	instrument4
	instrument5
	thermograph4
	thermograph5
	GroundStation2
)
(:init	
	(satellite satellite4)
	(instrument instrument4)
	(supports instrument4 thermograph4)
	(calibration_target instrument4 GroundStation2)
	(on_board instrument4 satellite4)
	(pointing satellite4 GroundStation2)
	(power_avail satellite4)
	
	(satellite satellite5)
	(instrument instrument5)
	(supports instrument5 thermograph5)
	(calibration_target instrument5 GroundStation2)
	(on_board instrument5 satellite5)
	(pointing satellite5 GroundStation2)
	(power_avail satellite5)
		
	(mode thermograph4)
	(mode thermograph5)
	
	(direction GroundStation2)
)
(:goal (and
	(power_on instrument4)
	(power_on instrument5)
))

)

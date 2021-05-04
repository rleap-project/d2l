(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite2	
	satellite3
	instrument2
	instrument3
	thermograph2
	thermograph3
	GroundStation2
)
(:init
	(satellite satellite2)
	(instrument instrument2)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 GroundStation2)
	(on_board instrument2 satellite2)
	(pointing satellite2 GroundStation2)
	(power_on instrument2)
	
	(satellite satellite3)
	(instrument instrument3)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 GroundStation2)
	(on_board instrument3 satellite3)
	(pointing satellite3 GroundStation2)
	(power_on instrument3)
		
	(mode thermograph2)
	(mode thermograph3)
	
	(direction GroundStation2)
)
(:goal (and
	(power_avail satellite2)
	(power_avail satellite3)
))

)

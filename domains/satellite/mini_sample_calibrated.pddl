(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite8	
	satellite9
	instrument8
	instrument9
	thermograph8
	thermograph9
	GroundStation2
	GroundStation5
)
(:init	
	(satellite satellite8)
	(instrument instrument8)
	(supports instrument8 thermograph8)
	(calibration_target instrument8 GroundStation2)
	(on_board instrument8 satellite8)
	(pointing satellite8 GroundStation5)
	(power_avail satellite8)
	
	(satellite satellite9)
	(instrument instrument9)
	(supports instrument9 thermograph9)
	(calibration_target instrument9 GroundStation2)
	(on_board instrument9 satellite9)
	(pointing satellite9 GroundStation5)
	(power_avail satellite9)
		
	(mode thermograph8)
	(mode thermograph9)
	
	(direction GroundStation2)
	(direction GroundStation5)
)
(:goal (and
	(calibrated instrument8)
	(calibrated instrument9)
))

)

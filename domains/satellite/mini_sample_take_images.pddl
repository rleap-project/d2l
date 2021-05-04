(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite10	
	instrument10
	instrument11
	thermograph10
	thermograph11
	GroundStation2
	GroundStation7
	Star6	
)
(:init	
	(satellite satellite10)
	(instrument instrument10)
	(instrument instrument11)
	(supports instrument10 thermograph10)
	(supports instrument11 thermograph11)
	(calibration_target instrument10 GroundStation2)
	(calibration_target instrument11 GroundStation2)
	(on_board instrument10 satellite10)
	(on_board instrument11 satellite10)	
	(pointing satellite10 GroundStation7)
	(calibrated instrument10)
	(power_on instrument10)
		
	(mode thermograph10)
	(mode thermograph11)
	
	(direction GroundStation2)
	(direction GroundStation7)
	(direction Star6)
)
(:goal (and
	(have_image Star6 thermograph10)
	(have_image Star6 thermograph11)
))

)

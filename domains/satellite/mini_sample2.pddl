(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0
	instrument0
	instrument1
	infrared0
	image2
	GroundStation2
	Planet3
	Phenomenon5
)
(:init
	(satellite satellite0)
	(instrument instrument0)
	(supports instrument0 infrared0)
	(calibration_target instrument0 GroundStation2)
	(instrument instrument1)
	(supports instrument1 image2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 GroundStation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Planet3)
	(mode infrared0)
	(mode image2)
	(direction GroundStation2)
	(direction Planet3)
	(direction Phenomenon5)
)
(:goal (and
	(have_image Planet3 infrared0)
	(have_image Phenomenon5 image2)
))

)

import pipelines
from sltp.util.misc import update_dict
#from sltp.util.names import visitall_names


def experiments():
    base = dict(
        domain_dir="parking-sequential-optimal",
        domain="domain.pddl",
        test_domain="domain.pddl",
        #feature_namer=visitall_names,
        pipeline="d2l_pipeline",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,

        distinguish_goals=True,
    )

    exps = dict()
    
    
    parking_base = update_dict(
		base,
		# Required info for latex table
		name = "parking",
        n_instances = 1,
        dimensions = "(X,X,X)",
    )

    exps["small"] = update_dict(
        parking_base,
        instances=[
            'instances/instance-1.pddl'            
        ],
        test_instances=[],
        test_policy_instances=all_test_instances(),

        max_concept_size=8,
        distance_feature_max_complexity=8,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        sampling_strategy="goal",
        verbosity=2,
    )

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        pipeline=pipelines.INCREMENTAL,
        instances=['instances/instance-1.pddl' ],
        test_policy_instances=all_test_instances(),
        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
        refine_policy_from_entire_sample=True,
        refinement_batch_size=1,
        compute_plan_on_flaws=True,
        num_random_walks=0,
        random_walk_length=0,
    )
    
    exps["small-sd2l"] = update_dict(
        base,
        instances=[
            'instances/instance-1.pddl'
        ],
        test_instances=[],
        test_policy_instances=all_test_instances(),
        acyclicity="sd2l",
        
        sampling_strategy="goal",
        n_features=2,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        #initial_sample_size=999999,
        #initial_sample_size=10,
        consistency_bound=10,
        optimal_steps=2,
        v_slack=2,
        verbosity=2,
        use_equivalence_classes=True,
    )

    return exps


def all_test_instances():
    return ["instance-{:02d}.pddl".format(i) for i in range(2, 20)]


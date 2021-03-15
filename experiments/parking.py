from sltp.util.misc import update_dict
#from sltp.util.names import visitall_names


def experiments():
    base = dict(
        domain_dir="parking-sequential-satisficing",
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

    exps["small"] = update_dict(
        base,
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


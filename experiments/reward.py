import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import reward_names, no_parameter


def experiments():
    base = dict(
        domain_dir="reward",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=reward_names,
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

    rew_base = update_dict(
    	base,
    	name = "rew",
    	n_instances = 1,
        dimensions = "5\\times 5",
    )

    exps["small"] = update_dict(
        rew_base,
        instances=["training_5x5.pddl"],
        # instances=["instance_5.pddl", "instance_4_blocked.pddl"],
        test_instances=[],
        test_policy_instances=all_test_instances(),
        
        max_concept_size=8,
        distance_feature_max_complexity=8,
        parameter_generator=no_parameter,
        # parameter_generator=None
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        sampling_strategy="goal",
        verbosity=2
    )

    exps["small-inc"] = update_dict(
        exps["small"],
        pipeline=pipelines.INCREMENTAL,
        instances=["instance_10x10_0.pddl","instance_10x10_1.pddl","instance_10x10_2.pddl","instance_10x10_3.pddl","instance_10x10_4.pddl"],
        validation_instances=["instance_10x10_0.pddl","instance_10x10_1.pddl","instance_10x10_2.pddl","instance_10x10_3.pddl","instance_10x10_4.pddl"],
        #instances=[f"prob{i:02d}.pddl" for i in range(3, 5)],
        #validation_instances=[f"prob{i:02d}.pddl" for i in range(3, 5)],
        test_policy_instances=all_test_instances(),
        verbosity=0,
    )
    
    exps["small-sd2l"] = update_dict(
        rew_base,
        instances=["training_5x5.pddl"],
        # instances=["instance_5.pddl", "instance_4_blocked.pddl"],
        test_instances=[],
        test_policy_instances=all_test_instances(),

        acyclicity="sd2l",

        sampling_strategy="goal",
		n_features=2,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        #initial_sample_size=999999,
        #initial_sample_size=10,
        consistency_bound=0,
        optimal_steps=2,
        v_slack=2,
        parameter_generator=no_parameter,
        verbosity=2,
        use_equivalence_classes=True,
    )
    
    exps["small-dtl"] = update_dict(
        rew_base,
        instances=["training_5x5.pddl"],
        # instances=["instance_5.pddl", "instance_4_blocked.pddl"],
        test_instances=[],
        test_policy_instances=all_test_instances(),

        acyclicity="dtl",

        sampling_strategy="goal",
		n_features=2,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        #initial_sample_size=999999,
        #initial_sample_size=10,
        consistency_bound=10,
        optimal_steps=2,
        v_slack=2,
        parameter_generator=no_parameter,
        verbosity=2,
        use_equivalence_classes=True,
    )
    

    exps["debug"] = update_dict(exps["small"], feature_generator=debug_features)

    # One reason for overfitting: in a 3x3 grid, with 2 booleans per dimension you can perfectly represent any position

    return exps


def debug_features(lang):
    unblocked_dist = "Dist[at;Restrict(adjacent,unblocked);reward]"
    nrewards = "Num[reward]"
    return [
        unblocked_dist,
        nrewards,
    ]


def all_test_instances():
    for gridsize in [5, 7, 10, 15, 20, 25]:
        for run in range(0, 5):
            yield f"instance_{gridsize}x{gridsize}_{run}.pddl"

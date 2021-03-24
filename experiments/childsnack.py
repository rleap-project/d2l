from sltp.util.misc import update_dict
from sltp.util.names import childsnack_names


def experiments():
    base = dict(
        domain_dir="childsnack-opt14-strips",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=childsnack_names,
        pipeline="d2l_pipeline",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,
    )

    exps = dict()
    
    child_base = update_dict(
		base,
		# Required info for latex table
		name = "child",
        n_instances = 2,
        dimensions = "(7,7,7,2,3,10)",
    )

    exps["small"] = update_dict(
        child_base,
        # instances=['sample{:02d}.pddl'.format(i) for i in range(1, 5)],
        instances=[
            # 'sample_mini.pddl',
            'sample01.pddl',
            'sample02.pddl',
            # 'sample03.pddl',

            # 'child-snack_pfile01-2.pddl'  # STATE SPACE TOO LARGE
        ],
        test_instances=[
            # 'child-snack_pfile01-2.pddl',
        ],
        test_policy_instances=all_test_instances(),
        
        # create_goal_features_automatically=True,
        max_concept_size=10,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        sampling_strategy="goal",
        verbosity=2
    )
    
    exps["small-sd2l"] = update_dict(
        child_base,
        # instances=['sample{:02d}.pddl'.format(i) for i in range(1, 5)],
        instances=[
            # 'sample_mini.pddl',
            'sample01.pddl',
            'sample02.pddl',
            'sample03.pddl',

            # 'child-snack_pfile01-2.pddl'  # STATE SPACE TOO LARGE
        ],
        test_instances=[
            # 'child-snack_pfile01-2.pddl',
        ],
        test_policy_instances=all_test_instances(),
        
		# Required info for latex table
        n_instances = 3,
        dimensions = "(8,8,8,2,3,11)",

		acyclicity="sd2l",
        sampling_strategy="goal",

        # create_goal_features_automatically=True,
        n_features=5,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        #initial_sample_size=999999,
        #initial_sample_size=10,
        consistency_bound=0,
        optimal_steps=2,
        v_slack=2,
        verbosity=2,
        use_equivalence_classes=True,
    )
    
    exps["small-dtl"] = update_dict(
        child_base,
        # instances=['sample{:02d}.pddl'.format(i) for i in range(1, 5)],
        instances=[
            # 'sample_mini.pddl',
            'sample01.pddl',
            'sample02.pddl',
            'sample03.pddl',

            # 'child-snack_pfile01-2.pddl'  # STATE SPACE TOO LARGE
        ],
        test_instances=[
            # 'child-snack_pfile01-2.pddl',
        ],
        test_policy_instances=all_test_instances(),

		# Required info for latex table
        n_instances = 3,
        dimensions = "(8,8,8,2,3,11)",
        
		acyclicity="dtl",
        sampling_strategy="goal",

        # create_goal_features_automatically=True,
        n_features=5,
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
    return ['child-snack_pfile01.pddl', 'child-snack_pfile01-2.pddl', 'child-snack_pfile02.pddl',
            'child-snack_pfile04-2.pddl', 'child-snack_pfile05.pddl', 'child-snack_pfile07-2.pddl',
            'child-snack_pfile08.pddl', 'child-snack_pfile10-2.pddl', 'child-snack_pfile01.pddl',
            'child-snack_pfile03-2.pddl', 'child-snack_pfile04.pddl', 'child-snack_pfile06-2.pddl',
            'child-snack_pfile07.pddl', 'child-snack_pfile09-2.pddl', 'child-snack_pfile10.pddl',
            'child-snack_pfile02-2.pddl', 'child-snack_pfile03.pddl', 'child-snack_pfile05-2.pddl',
            'child-snack_pfile06.pddl', 'child-snack_pfile08-2.pddl', 'child-snack_pfile09.pddl']

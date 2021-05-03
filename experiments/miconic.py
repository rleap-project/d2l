import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import miconic_names


def experiments():
    base = dict(
        domain_dir="miconic",
        # domain="domain.pddl",
        # test_domain="domain.pddl",

        # Without the fix, the "board" action allows to board passengers that are not on the floor anymore!
        test_domain="domain-with-fix.pddl",
        domain="domain-with-fix.pddl",

        feature_namer=miconic_names,
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

    micon_base = update_dict(
    	base,
    	name = "micon",
    	n_instances = 2,
    	dimensions = "(4,8)",
    )
    
    exps["small"] = update_dict(
        micon_base,
        instances=[
            # 's2-0.pddl',
            # 's2-1.pddl',
            # 's2-2.pddl',
            # 's2-3.pddl',
            # 's3-0.pddl',
            's4-0.pddl',
            'training2.pddl',
        ],
        test_instances=[
        ],
        test_policy_instances=all_test_instances(),
        
        max_concept_size=8,
        distance_feature_max_complexity=8,

        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,


        initial_sample_size=100,
        # acyclicity="asp",
        verbosity=1,
        sampling_strategy="goal"
    )

    exps["small-orig-inc"] = update_dict(
        exps["small"],
        pipeline=pipelines.INCREMENTAL,
        instances=[ 's4-0.pddl', 'training2.pddl',],
        validation_instances=[ 's4-0.pddl', 'training2.pddl',],
        test_policy_instances=all_test_instances(),
        
        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
        refine_policy_from_entire_sample=True,
        refinement_batch_size=2,
        compute_plan_on_flaws=True,
        num_random_walks=2,
        random_walk_length=10,
    )

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        pipeline=pipelines.INCREMENTAL,
        instances=["s5-0.pddl","s7-0.pddl"],
        #instances=["s7-0.pddl"],
        test_policy_instances=all_test_instances(),
        
        #sampling_strategy="full",
        #initial_sample_size=999999,
        #verbosity=2,
        #refine_policy_from_entire_sample=True,
        #refinement_batch_size=2,
        #compute_plan_on_flaws=True,
        #num_random_walks=2,
        #random_walk_length=10,
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
        micon_base,
        instances=[
            # 's2-0.pddl',
            # 's2-1.pddl',
            # 's2-2.pddl',
            # 's2-3.pddl',
            # 's3-0.pddl',
            #'s4-0.pddl',
            #'training2.pddl'#, #(updated with a new origin passenger)
            's7-0.pddl'
        ],
        test_instances=[
        ],
        test_policy_instances=all_test_instances(),
        
    	n_instances = 1,
    	dimensions = "(7,14)",

        acyclicity="sd2l",
        sampling_strategy="goal",
		n_features=4,
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
        micon_base,
        instances=[
            # 's2-0.pddl',
            # 's2-1.pddl',
            # 's2-2.pddl',
            # 's2-3.pddl',
            # 's3-0.pddl',
            #'s4-0.pddl',
            #'training2.pddl'#, #(updated with a new origin passenger)
            's7-0.pddl'
        ],
        test_instances=[
        ],
        test_policy_instances=all_test_instances(),
        
    	n_instances = 1,
    	dimensions = "(7,14)",

        acyclicity="dtl",
        sampling_strategy="goal",
		n_features=4,
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
    
    

    exps["small2"] = update_dict(
        exps["small"],
        instances=[
            's3-0.pddl',
            'training2.pddl',
        ],
    )

    exps["debug"] = update_dict(
        exps["small"],
        instances=[
            # 'debug.pddl',
            's4-0.pddl',
            'training2.pddl',
        ],

        feature_generator=debug_features,
        # d2l_policy=debug_policy,
        print_denotations=True,

        test_policy_instances=all_test_instances(),
        # test_policy_instances=[
        #     's4-0.pddl',
        # ]
    )

    return exps


def all_test_instances():
    instances = []
    for i in range(1, 31, 3):  # jump 3-by-3 to have fewer instances
        for j in range(0, 5):  # Each x has 5 subproblems
            instances.append("s{}-{}.pddl".format(i, j))
    return instances


nserved = "Num[served]"  # k = 1
nboarded = "Num[boarded]"  # k = 1
lift_at_dest_some_boarded_pass = "Bool[And(lift-at,Exists(Inverse(destin),boarded))]"  # k = 6

# Note that this one below appears correct but is not enough, since it doesn't allow us to express
# that moving between floors without having picked all passengers in previous floor is not good.
n_pass_ready_to_board = "Num[And(And(Not(boarded),Not(served)),Exists(origin,lift-at))]"
# This one is more complex, but better (k = 10)
lift_at_origin_some_awaiting_pass = "Bool[And(lift-at,Exists(Inverse(origin),And(Not(boarded),Not(served))))]"


def debug_features(lang):
    return [
        nserved,
        nboarded,
        lift_at_origin_some_awaiting_pass,
        lift_at_dest_some_boarded_pass,
    ]


def debug_policy():
    return [
        # Decreasing the # boarded passengers is always good (because they become served)
        [(nboarded, 'DEC')],

        # Boarding people is always good
        [(nboarded, 'INC')],

        # Moving to a floor with unserved people is good as long as we leave no unboarded passenger in current floor
        # (i.e. we don't want that lift_at_origin_some_awaiting_pass NILs)
        [(lift_at_origin_some_awaiting_pass, 'ADD')],

        # Moving to the destination floor of some boarded pass is good, as long as we leave no pass waiting
        [(lift_at_origin_some_awaiting_pass, "=0"), (lift_at_dest_some_boarded_pass, 'ADD')],
    ]

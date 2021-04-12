import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import spanner_names


def experiments():
    base = dict(
        domain_dir="spanner-ipc11-learning",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=spanner_names,
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

    span_base = update_dict(
    	base,
    	name = "span",
    	n_instances = 3,
    	dimensions = "(6,10)",
    )

    exps["small"] = update_dict(
        span_base,
        pipeline="d2l_pipeline",
        instances=[
            "prob-2-2-10.pddl",
            "prob-4-2-5.pddl",
            "prob-6_4_10.pddl",

        ],
        test_instances=[
        ],
        test_policy_instances=[
            "prob-10-10-10-1540903568.pddl",
            "prob-15-10-8-1540913795.pddl"
        ] + all_test_instances(),
        
        max_concept_size=8,
        distance_feature_max_complexity=8,

        # d2l_policy=debug_policy
        # comparison_features=True,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        sampling_strategy="goal",
        verbosity=2
    )

    exps["small-inc"] = update_dict(
        exps["small"],
        pipeline=pipelines.INCREMENTAL,
        instances=[
            'pfile02-006.pddl',
        ],
        validation_instances=[
            'pfile02-006.pddl',
        ],
        test_policy_instances=[
            "prob-10-10-10-1540903568.pddl",
            "prob-15-10-8-1540913795.pddl"
        ] + all_test_instances(),
        verbosity=0,
    )

    exps["small-sd2l"] = update_dict(
        span_base,
        pipeline="d2l_pipeline",
        instances=[
            "prob-2-2-10.pddl",
            "prob-4-2-5.pddl",
            "prob-6_4_10.pddl",

        ],
        test_instances=[
        ],
        test_policy_instances=[
            "prob-10-10-10-1540903568.pddl",
            "prob-15-10-8-1540913795.pddl"
        ] + all_test_instances(),

        acyclicity="sd2l",
        sampling_strategy="goal",
        
        n_features=3,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        #initial_sample_size=999999,
        #initial_sample_size=10,
        v_slack=2,
        consistency_bound=0,
        optimal_steps=2,
        verbosity=2,
        use_equivalence_classes=True,
    )

    exps["small-dtl"] = update_dict(
        span_base,
        pipeline="d2l_pipeline",
        instances=[
            "prob-2-2-10.pddl",
            "prob-4-2-5.pddl",
            "prob-6_4_10.pddl",

        ],
        test_instances=[
        ],
        test_policy_instances=[
            "prob-10-10-10-1540903568.pddl",
            "prob-15-10-8-1540913795.pddl"
        ] + all_test_instances(),

        acyclicity="dtl",
        sampling_strategy="goal",
        
        n_features=3,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        #initial_sample_size=999999,
        #initial_sample_size=10,
        v_slack=2,
        consistency_bound=10,
        optimal_steps=2,
        verbosity=2,
        use_equivalence_classes=True,
    )
    

    return exps


def debug_policy():
    n_carried = "Num[Exists(Inverse(carrying),<universe>)]"  # K=3
    n_unreachable_locs = "Num[Exists(Star(link),Exists(Inverse(at),man))]"  # K=7
    n_tightened = "Num[tightened]"  # K=1
    n_spanners_same_loc_as_bob = "Num[And(Exists(at,Exists(Inverse(at),man)),spanner)]"  # K=8
    not_carrying_enough_spanners = "LessThan{Num[Exists(Inverse(carrying),<universe>)]}{Num[loose]}"  # K=5

    return [
        # picking a spanner when bob doesn't carry enough spanners is good:
        [(not_carrying_enough_spanners, ">0"), (n_carried, 'INC')],

        # Tightening a nut when possible is always good:
        [(n_tightened, 'INC')],

        # Moving to the right when bob already carries enough spanners is good:
        [(not_carrying_enough_spanners, "=0"), (n_unreachable_locs, 'INC')],

        # Moving to the right when there's no more spanners in same location as bob is good:
        [(n_spanners_same_loc_as_bob, "=0"), (n_unreachable_locs, 'INC')],
    ]


def all_test_instances():
    import math
    return [f"pfile0{math.ceil(i/5)}-0{i:02d}.pddl" for i in range(1, 31)]

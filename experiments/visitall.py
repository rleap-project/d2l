import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import visitall_names


def experiments():
    base = dict(
        domain_dir="visitall-opt11-strips",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=visitall_names,
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

    visit_base = update_dict(
        base,
        name="visit",
        n_instances=1,
        dimensions="3\\times 3",
    )

    exps["small"] = update_dict(
        visit_base,
        instances=[
            'problem03-full.pddl',
            # 'problem04-full.pddl',
            # 'problem05-full.pddl',
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

    exps["small-orig-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=['problem03-full.pddl'],
        validation_instances=['problem03-full.pddl'],
        # instances=["problem{:02d}-full.pddl".format(i) for i in range(2, 6)],
        # validation_instances=["problem{:02d}-full.pddl".format(i) for i in range(2, 6)],
        test_policy_instances=["problem{:02d}-full.pddl".format(i) for i in range(6, 12)],

        refine_policy_from_entire_sample=False,
        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
    )

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=['problem05-full.pddl'],
        validation_instances=['problem05-full.pddl'],
        # instances=["problem{:02d}-full.pddl".format(i) for i in range(2, 6)],
        # validation_instances=["problem{:02d}-full.pddl".format(i) for i in range(2, 6)],
        test_policy_instances=["problem{:02d}-full.pddl".format(i) for i in range(6, 12)],

        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
        refine_policy_from_entire_sample=True,
        refinement_batch_size=10,
        compute_plan_on_flaws=True,
    )
    
    exps["small-sd2l"] = update_dict(
        visit_base,
        instances=[
            'problem03-full.pddl',
            # 'problem04-full.pddl',
            # 'problem05-full.pddl',
        ],
        test_instances=[],
        test_policy_instances=all_test_instances(),
        acyclicity="sd2l",

        sampling_strategy="goal",
        n_features=2,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        # initial_sample_size=999999,
        # initial_sample_size=10,
        consistency_bound=0,
        optimal_steps=2,
        v_slack=2,
        verbosity=2,
        use_equivalence_classes=True,
    )

    exps["small-dtl"] = update_dict(
        visit_base,
        instances=[
            'problem03-full.pddl',
            # 'problem04-full.pddl',
            # 'problem05-full.pddl',
        ],
        test_instances=[],
        test_policy_instances=all_test_instances(),
        acyclicity="dtl",

        sampling_strategy="goal",
        n_features=2,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        # initial_sample_size=999999,
        # initial_sample_size=10,
        consistency_bound=10,
        optimal_steps=2,
        v_slack=2,
        verbosity=2,
        use_equivalence_classes=True,
    )

    exps["debug"] = update_dict(
        base,
        instances=[
            # 'problem02-full.pddl',
            # 'problem03-full.pddl',
            "tests.pddl",
        ],
        test_instances=[
            # 'problem03-full.pddl',
            # 'problem04-full.pddl',
        ],
        # test_policy_instances=all_test_instances(),

        # max_concept_size=8,
        # distance_feature_max_complexity=8,
        # cond_feature_max_complexity=8,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        print_denotations=True,

        feature_generator=debug_features,
    )

    return exps


def all_test_instances():
    return ["problem{:02d}-full.pddl".format(i) for i in range(2, 12)]


def debug_features(lang):
    dist_to_unvisited = "Dist[at-robot;connected;Not(visited)]"
    num_visited = "Num[visited]"
    # num_unvisited = "Num[Not(visited)]"
    return [
        dist_to_unvisited,
        num_visited,
        # num_unvisited,
    ]

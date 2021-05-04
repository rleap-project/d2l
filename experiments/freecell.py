import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import gripper_names, gripper_parameters


def experiments():
    base = dict(
        domain_dir="freecell",
        domain="domain.pddl",
        test_domain="domain.pddl",
        # feature_namer=gripper_names,
        pipeline="d2l_pipeline",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,
        distinguish_goals=True,

        name="grip",
        n_instances=1,
        dimensions="4",
    )

    exps = dict()

    exps["small"] = update_dict(
        base,
        pipeline=pipelines.INCREMENTAL,
        instances=["p01.pddl"],
        # instances=["p02.pddl"],
        test_instances=[],
        test_policy_instances=["p01.pddl"],

        max_concept_size=8,
        distance_feature_max_complexity=8,

        # parameter_generator=gripper_parameters,  # Works also, but no real advantage
        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        # print_hstar_in_feature_matrix=True,
        sampling_strategy="goal",
        verbosity=2,

        num_random_rollouts=10,
        random_walk_length=20,
    )

    return exps

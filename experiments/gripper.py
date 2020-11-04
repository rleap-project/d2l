from sltp.util.misc import update_dict
from sltp.util.names import gripper_names, gripper_parameters


def experiments():
    base = dict(
        domain_dir="gripper",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=gripper_names,
        pipeline="transition_classifier",
        maxsat_encoding="separation",
        complete_only_wrt_optimal=True,
        prune_redundant_states=False,
        optimal_selection_strategy="complete",
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
        # instances=["sample-2balls.pddl", "sample-small.pddl"],
        instances=["prob01.pddl"],
        # test_instances=[f"prob{i:02d}.pddl" for i in range(3, 11)],
        test_instances=[],
        test_policy_instances=[f"prob{i:02d}.pddl" for i in range(3, 21)],

        max_concept_size=8,
        distance_feature_max_complexity=8,

        # parameter_generator=gripper_parameters,  # Works also, but no real advantage
        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        use_incremental_refinement=True,
    )

    return exps

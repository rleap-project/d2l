import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import gripper_names, gripper_parameters


def experiments():
    base = dict(
        domain_dir="maintenance-small",
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
    )

    exps = dict()

    grip_base = update_dict(
        base,
        name="grip",
        n_instances=1,
        dimensions="4",
    )

    exps["small"] = update_dict(
        grip_base,
        instances=["maintenance-3-4-3.pddl"],
        # test_instances=[f"prob{i:02d}.pddl" for i in range(3, 11)],
        test_instances=[],
        test_policy_instances=["maintenance-3-4-3.pddl"],

        max_concept_size=8,
        distance_feature_max_complexity=8,

        # parameter_generator=gripper_parameters,  # Works also, but no real advantage
        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        # print_hstar_in_feature_matrix=True,
        sampling_strategy="goal",
        verbosity=2
    )

    return exps

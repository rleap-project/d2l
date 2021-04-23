import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import gripper_names, gripper_parameters


def experiments():
    base = dict(
        domain_dir="gripper",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=gripper_names,
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
        # print_hstar_in_feature_matrix=True,
        sampling_strategy="goal",
        verbosity=2
    )

    exps["small-orig-inc"] = update_dict(
        exps["small"],
        pipeline=pipelines.INCREMENTAL,
        instances=["prob01.pddl"],
        validation_instances=["prob01.pddl"],
        # instances=[f"prob{i:02d}.pddl" for i in range(3, 5)],
        # validation_instances=[f"prob{i:02d}.pddl" for i in range(3, 5)],
        test_policy_instances=[f"prob{i:02d}.pddl" for i in range(5, 21)],
        verbosity=2,
        refine_policy_from_entire_sample=False,
        initial_sample_size=999999,
        sampling_strategy="full"

    )

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        pipeline=pipelines.INCREMENTAL,
        instances=["prob04-fixed.pddl","prob05.pddl"],
        validation_instances=["prob04-fixed.pddl","prob05.pddl"],
        # instances=[f"prob{i:02d}.pddl" for i in range(3, 5)],
        # validation_instances=[f"prob{i:02d}.pddl" for i in range(3, 5)],
        test_policy_instances=["sample-tiny.pddl"]+[f"prob{i:02d}.pddl" for i in range(5, 21)],
        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
        refine_policy_from_entire_sample=True,
        refinement_batch_size=10,
        compute_plan_on_flaws=True,

    )

    exps["small-sd2l"] = update_dict(
        grip_base,
        # instances=["sample-2balls.pddl", "sample-small.pddl"],
        instances=["prob01.pddl"],
        # test_instances=[f"prob{i:02d}.pddl" for i in range(3, 11)],
        test_instances=[],
        test_policy_instances=[f"prob{i:02d}.pddl" for i in range(3, 21)],

        acyclicity="sd2l",

        sampling_strategy="goal",
        n_features=3,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        v_slack=2,
        # initial_sample_size=10,
        # initial_sample_size=999999,
        consistency_bound=0,
        optimal_steps=2,
        # parameter_generator=gripper_parameters,  # Works also, but no real advantage
        verbosity=2,
        use_equivalence_classes=True,
    )

    exps["small-dtl"] = update_dict(
        grip_base,
        # instances=["sample-2balls.pddl", "sample-small.pddl"],
        instances=["prob01.pddl"],
        # test_instances=[f"prob{i:02d}.pddl" for i in range(3, 11)],
        test_instances=[],
        test_policy_instances=[f"prob{i:02d}.pddl" for i in range(3, 21)],

        acyclicity="dtl",

        sampling_strategy="goal",
        n_features=3,
        max_concept_size=8,
        distance_feature_max_complexity=8,
        v_slack=2,
        # initial_sample_size=10,
        # initial_sample_size=999999,
        consistency_bound=10,
        optimal_steps=2,
        # parameter_generator=gripper_parameters,  # Works also, but no real advantage
        verbosity=2,
        use_equivalence_classes=True,
    )

    return exps

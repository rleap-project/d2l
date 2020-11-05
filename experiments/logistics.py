from sltp.util.misc import update_dict
from sltp.util.names import logistics_names


def experiments():
    base = dict(
        # domain_dir="gripper-m",
        domain_dir="logistics98",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=logistics_names,
        pipeline="d2l_pipeline",
        maxsat_encoding="d2l",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,

        distinguish_goals=True,
    )

    exps = dict()

    # Goal: arbitrary logistics goal
    exps["small"] = update_dict(
        base,
        instances=[f'sample{i}.pddl' for i in [2]],
        # test_instances=["prob{:02d}.pddl".format(i) for i in range(2, 5)],
        test_instances=[],
        test_policy_instances=all_instances(),

        distance_feature_max_complexity=8,
        max_concept_size=8,

        use_equivalence_classes=True,
        # use_feature_dominance=True,
        use_incremental_refinement=True,
    )

    return exps


def all_instances():
    return [f"prob0{i}.pddl" for i in range(1, 3)]

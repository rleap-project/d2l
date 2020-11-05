
from sltp.util.misc import update_dict
from sltp.util.names import gridworld_names


def experiments():
    base = dict(
        domain_dir="gridworld",
        domain="domain_strips.pddl",
        test_domain="domain_strips.pddl",
        feature_namer=gridworld_names,
        pipeline="d2l_pipeline",
        maxsat_encoding="d2l",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,
    )

    exps = dict()

    # 3x3 overfits: in a 3x3 grid, with 2 booleans per dimension you can represent the position

    exps["5x5"] = update_dict(
        base,

        instances=[
            "instance_strips_5.pddl",
            "instance_strips_5_not_corner.pddl",
            "instance_strips_5_not_corner_2.pddl",
        ],
        test_instances=[],
        test_policy_instances=[],

        max_concept_size=8,
        distance_feature_max_complexity=8,

        # parameter_generator=gripper_parameters,  # Works also, but no real advantage
        parameter_generator=add_domain_parameters_strips,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        # use_incremental_refinement=True,

        # feature_generator=generate_features_1,
    )

    return exps


def add_domain_parameters(language):
    # language.constant(2, "coordinate")  # x-goal coordinate
    # language.constant(3, "coordinate")  # x-goal coordinate
    # language.constant(10, "coordinate")  # grid limits!!
    # [language.constant(i, "coordinate") for i in range(1, 11)]
    return [language.constant(1, "coordinate")]  # grid limits!!


def add_domain_parameters_strips(language):
    return []


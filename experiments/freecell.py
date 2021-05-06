import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import freecell_names


def experiments():
    base = dict(
        domain_dir="freecell",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=freecell_names,
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
        pipeline=pipelines.MILESTONES,
        instances=["p01.pddl"],
        # instances=["p02.pddl"],
        test_instances=[],
        test_policy_instances=["p01.pddl"],

        max_concept_size=6,
        distance_feature_max_complexity=6,

        # parameter_generator=gripper_parameters,  # Works also, but no real advantage
        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        # print_hstar_in_feature_matrix=True,
        sampling_strategy="goal",
        verbosity=2,

        num_random_rollouts=20,
        random_walk_length=10,
    )

    return exps

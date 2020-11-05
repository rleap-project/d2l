from sltp.util.misc import update_dict
from sltp.util.names import satellite_names


def experiments():
    base = dict(
        domain_dir="satellite",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=satellite_names,
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

    exps["small"] = update_dict(
        base,
        instances=[
            'p01-pfile1.pddl',
            'p02-pfile2.pddl',
        ],
        test_instances=[],
        test_policy_instances=all_test_instances(),

        max_concept_size=8,
        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        # use_incremental_refinement=True,
    )

    return exps


def all_test_instances():
    return ["p{:02d}-pfile{}.pddl".format(i, i) for i in range(1, 21)]


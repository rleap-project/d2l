from sltp.util.misc import update_dict
from sltp.util.names import barman_names


def experiments():
    base = dict(
        domain_dir="barman-opt11-strips",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=barman_names,
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
            'sample01.pddl',
        ],
        test_instances=[
            # 'child-snack_pfile01-2.pddl',
        ],
        test_policy_instances=all_test_instances(),

        max_concept_size=8,
        distance_feature_max_complexity=8,

        use_equivalence_classes=True,
        # use_feature_dominance=True,
        use_incremental_refinement=True,
    )

    return exps


def all_test_instances():
    return ['pfile01-001.pddl']

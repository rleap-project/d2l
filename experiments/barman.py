import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import barman_names


def experiments():
    base = dict(
        domain_dir="barman-opt11-strips",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=barman_names,
        pipeline="d2l_pipeline",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,
    )

    exps = dict()

    barman_base = update_dict(
        base,
        name="barman",
        n_instances=1,
        dimensions="(X,X,X)",
    )

    exps["small"] = update_dict(
        barman_base,
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
    )

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=['sample01.pddl'],
        validation_instances=['sample01.pddl'],
        test_policy_instances=all_test_instances(),

        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
        refine_policy_from_entire_sample=False,
        refinement_batch_size=20,
    )

    return exps


def all_test_instances():
    return ['pfile01-001.pddl']

import pipelines
from sltp.util.misc import update_dict



def experiments():
    base = dict(
        domain_dir="grid",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=None,
        pipeline="d2l_pipeline",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,
    )

    exps = dict()

    grid_base = update_dict(
        base,
        name="grid",
        n_instances=2,
        dimensions="(X,X,X)",
    )

    exps["small"] = update_dict(
        grid_base,
        instances=[
            'prob01.pddl',
            'prob02.pddl',
        ],
        test_instances=[],
        test_policy_instances=all_test_instances(),

        max_concept_size=8,
        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
    )

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=[
            'prob01.pddl',
            'prob02.pddl',
        ],
        validation_instances=[
            'prob01.pddl',
            'prob02.pddl',
        ],
        test_policy_instances=all_test_instances(),

        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
        refine_policy_from_entire_sample=False,
        refinement_batch_size=20,
    )

    return exps


def all_test_instances():
    return ["prob{:02d}.pddl".format(i) for i in range(3, 6)]


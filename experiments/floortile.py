import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import floortile_names


def experiments():
    base = dict(
        domain_dir="floortile-opt11-strips",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=floortile_names,
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

    floortile_base = update_dict(
        base,
        name="floortile",
        n_instances=1,
        dimensions="(X,X,X)",
    )

    exps["small"] = update_dict(
        floortile_base,
        instances=["training1.pddl"],
        # instances=["testing.pddl"],
        test_instances=[],
        test_policy_instances=["opt-p01-002.pddl"],

        max_concept_size=6,
        distance_feature_max_complexity=6,

        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
    )

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=["training1.pddl"],
        test_policy_instances=["opt-p01-002.pddl"],

        #sampling_strategy="full",
        #initial_sample_size=999999,
        #verbosity=2,
        #refine_policy_from_entire_sample=True,
        #refinement_batch_size=2,
        #compute_plan_on_flaws=True,
        #num_random_walks=2,
        #random_walk_length=10,
        sampling_strategy="full",
        initial_sample_size=999999,
        verbosity=2,
        refine_policy_from_entire_sample=True,
        refinement_batch_size=1,
        compute_plan_on_flaws=True,
        num_random_walks=0,
        random_walk_length=0,
    )

    return exps

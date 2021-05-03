import pipelines
from sltp.util.misc import update_dict
from sltp.util.names import satellite_names


def experiments():
    base = dict(
        domain_dir="satellite",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=satellite_names,
        pipeline="d2l_pipeline",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,
    )

    exps = dict()

    satellite_base = update_dict(
        base,
        name="satellite",
        n_instances=2,
        dimensions="(X,X,X)",
    )

    exps["small"] = update_dict(
        satellite_base,
        instances=[
            'p01-pfile1.pddl',
            'p02-pfile2.pddl',
        ],
        test_instances=[],
        test_policy_instances=all_test_instances(),

        max_concept_size=12,
        distance_feature_max_complexity=16,
        parameter_generator=None,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
    )

    exps["small-orig-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=[
            #'mini_sample1.pddl',
            'mini_sample2.pddl',
        ],
        test_policy_instances=all_test_instances(),

        #sampling_strategy="full",
        #initial_sample_size=999999,
        #verbosity=2,
        #refine_policy_from_entire_sample=True,
        #refinement_batch_size=2,
        #compute_plan_on_flaws=True,
        #num_random_walks=10,
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

    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=[
            'p01-pfile1.pddl',
            'p02-pfile2.pddl',
        ],
        test_policy_instances=all_test_instances(),

        #sampling_strategy="full",
        #initial_sample_size=999999,
        #verbosity=2,
        #refine_policy_from_entire_sample=True,
        #refinement_batch_size=1,
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


def all_test_instances():
    return ["p{:02d}-pfile{}.pddl".format(i, i) for i in range(1, 21)]


import pipelines
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
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,

        distinguish_goals=True,
    )

    exps = dict()

    logistics_base = update_dict(
        base,
        name="logistics",
        n_instances=2,
        dimensions="(X,X,X)",
    )

    # Goal: arbitrary logistics goal
    exps["small"] = update_dict(
        logistics_base,
        instances=[f'sample{i}.pddl' for i in [2]],
        # test_instances=["prob{:02d}.pddl".format(i) for i in range(2, 5)],
        test_instances=[],
        test_policy_instances=all_instances(),

        distance_feature_max_complexity=14,
        max_concept_size=9,
        use_equivalence_classes=True,
        sampling_strategy="goal",
        verbosity=2,
    )

    exps["small-orig-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=[f'sample{i}.pddl' for i in [2]],
        test_policy_instances=all_instances(),
        
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
    
    exps["small-ipc-inc"] = update_dict(
        exps["small"],
        distinguish_goals=True,
        pipeline=pipelines.INCREMENTAL,
        instances=[f'sample{i}.pddl' for i in [1,2,3]],
        test_policy_instances=all_instances(),
        
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
    
    exps["small-sd2l"] = update_dict(
        base,
        instances=[f'sample{i}.pddl' for i in [2]],
        # test_instances=["prob{:02d}.pddl".format(i) for i in range(2, 5)],
        test_instances=[],
        test_policy_instances=all_instances(),
        acyclicity="sd2l",
        
        sampling_strategy="goal",
        n_features=5,
        max_concept_size=6,
        distance_feature_max_complexity=6,
        #initial_sample_size=999999,
        #initial_sample_size=10,
        consistency_bound=0,
        optimal_steps=0,
        v_slack=2,
        verbosity=2,
        use_equivalence_classes=True,
    )

    return exps


def all_instances():
    return [f"prob0{i}.pddl" for i in range(1, 36)]

from sltp.util.misc import update_dict
from sltp.util.names import reward_names, no_parameter


def experiments():
    base = dict(
        domain_dir="reward",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=reward_names,
        pipeline="transition_classifier",
        maxsat_encoding="separation",
        complete_only_wrt_optimal=True,
        prune_redundant_states=False,
        optimal_selection_strategy="complete",
        num_states="all",
        concept_generator=None,
        parameter_generator=None,
        v_slack=2,

        # concept_generation_timeout=120,  # in seconds
        maxsat_timeout=None,

        distinguish_goals=True,
    )

    exps = dict()

    exps["small"] = update_dict(
        base,
        instances=["training_5x5.pddl"],
        # instances=["instance_5.pddl", "instance_4_blocked.pddl"],
        test_instances=[],
        test_policy_instances=all_test_instances(),

        max_concept_size=8,
        distance_feature_max_complexity=8,
        parameter_generator=no_parameter,
        # parameter_generator=None
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        use_incremental_refinement=True,
    )

    exps["debug"] = update_dict(exps["small"], feature_generator=debug_features)

    # One reason for overfitting: in a 3x3 grid, with 2 booleans per dimension you can perfectly represent any position
    # exps['sample_2x2_1reward'] = update_dict(exps['sample_1x3'], instances=["sample_2x2_1reward.pddl"])
    # exps['sample_2x2_2rewards'] = update_dict(exps['sample_1x3'], instances=["sample_2x2_2rewards.pddl"])
    # exps['sample_3x3_2rewards'] = update_dict(exps['sample_1x3'], instances=["sample_3x3_2rewards.pddl"])
    # exps['instance_5'] = update_dict(exps['sample_1x3'], instances=["instance_5.pddl", "instance_4_blocked.pddl"])
    # exps['instance_5_no_marking'] = update_dict(exps['instance_5'], complete_only_wrt_optimal=False,)

    # Same but using goal-concepts instead of goal parameters:
    # exps["instance_5_gc"] = update_dict(exps["instance_5"], parameter_generator=None)

    return exps


def debug_features(lang):
    unblocked_dist = "Dist[at;Restrict(adjacent,unblocked);reward]"
    nrewards = "Num[reward]"
    return [
        unblocked_dist,
        nrewards,
    ]


def all_test_instances():
    for gridsize in [5, 7, 10, 15, 20, 25]:
        for run in range(0, 5):
            yield f"instance_{gridsize}x{gridsize}_{run}.pddl"

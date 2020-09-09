from sltp.util.misc import update_dict
from sltp.util.names import spanner_names


def experiments():
    base = dict(
        # domain_dir="gripper-m",
        domain_dir="spanner-small",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=spanner_names,
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

        force_zeros=True,
        distinguish_goals=True,
    )

    exps = dict()

    exps["small"] = update_dict(
        base,
        pipeline="transition_classifier",
        instances=[
            "prob-3-2-2.pddl",
            "prob-3-2-2_b.pddl",
            "prob-3-2-2_c.pddl",
            "prob-3-2-2_d.pddl",
            "prob-4-2-5.pddl",
            "prob-6_4_10.pddl",
            "prob-2-2-10.pddl",
            "prob-2-2-10_b.pddl"

        ],
        test_instances=[
            "prob-4-3-3-1540907466.pddl",
            "prob-4-4-3-1540907456.pddl",
            "prob-4-2-6.pddl",
            "prob-4-2-7.pddl",
            # "prob-6-6-10.pddl"
        ],
        test_policy_instances=[
            "prob-10-10-10-1540903568.pddl",
            "prob-15-10-8-1540913795.pddl"
        ],

        max_concept_size=8,
        distance_feature_max_complexity=8,

        # transition_classification_policy=debug_policy
        # comparison_features=True,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        use_incremental_refinement=True,
    )

    return exps


def debug_policy():
    n_carried = "Num[Exists(Inverse(carrying),<universe>)]"  # K=3
    n_unreachable_locs = "Num[Exists(Star(link),Exists(Inverse(at),man))]"  # K=7
    n_tightened = "Num[tightened]"  # K=1
    n_spanners_same_loc_as_bob = "Num[And(Exists(at,Exists(Inverse(at),man)),spanner)]"  # K=8
    not_carrying_enough_spanners = "LessThan{Num[Exists(Inverse(carrying),<universe>)]}{Num[loose]}"  # K=5

    return [
        # picking a spanner when bob doesn't carry enough spanners is good:
        [(not_carrying_enough_spanners, ">0"), (n_carried, 'INC')],

        # Tightening a nut when possible is always good:
        [(n_tightened, 'INC')],

        # Moving to the right when bob already carries enough spanners is good:
        [(not_carrying_enough_spanners, "=0"), (n_unreachable_locs, 'INC')],

        # Moving to the right when there's no more spanners in same location as bob is good:
        [(n_spanners_same_loc_as_bob, "=0"), (n_unreachable_locs, 'INC')],
    ]

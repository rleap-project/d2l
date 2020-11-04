from sltp.util.misc import update_dict
from sltp.util.names import hanoi_names


def experiments():

    base = dict(
        domain_dir="hanoi",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=hanoi_names,
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
    )

    exps = dict()

    exps["small"] = update_dict(
        base,
        instances=[
            # 'p01.pddl',
            # 'p02.pddl',
            'p03.pddl',
            'p04.pddl',
            'p05.pddl',
        ],

        test_policy_instances=[
            'p04.pddl',
            'p05.pddl',
            'p06.pddl',
            'p07.pddl',
            'p08.pddl',
        ],

        max_concept_size=8,
        use_equivalence_classes=True,
        # use_feature_dominance=True,
        use_incremental_refinement=True,
    )

    return exps

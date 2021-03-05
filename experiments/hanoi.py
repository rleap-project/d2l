from sltp.util.misc import update_dict
from sltp.util.names import hanoi_names


def experiments():

    base = dict(
        domain_dir="hanoi",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=hanoi_names,
        pipeline="d2l_pipeline",
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
    )

    exps["small-sd2l"] = update_dict(
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
        
        acyclicity="sd2l",

		n_features=4,
        sampling_strategy="goal",
        max_concept_size=8,
        distance_feature_max_complexity=8,
        #initial_sample_size=10,
        consistency_bound=10,
        optimal_steps=2,
        v_slack=2,
        verbosity=2,
        use_equivalence_classes=True,
    )

    return exps

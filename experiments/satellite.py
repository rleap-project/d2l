from sltp.util.misc import update_dict


def experiments():

    base = dict(
        domain_dir="satellite",
        domain="domain.pddl",
        test_domain="domain.pddl",
        complete_only_wrt_optimal=True
    )

    exps = dict()

    exps["p1"] = update_dict(
        base,
        instances=[
            'p01-pfile1.pddl',
        ],
        test_instances=[
            'p05-pfile5.pddl',
        ],
        num_states=200000,
        num_tested_states=50000,
        num_sampled_states=None,  # Take all expanded states into account
        initial_concept_bound=8, max_concept_bound=16, concept_bound_step=1,
        concept_generator=None,
        parameter_generator=None,
    )

    return exps

from sltp.util.misc import update_dict
from sltp.util.names import logistics_names


def experiments():
    base = dict(
        # domain_dir="gripper-m",
        domain_dir="logistics98",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=logistics_names,
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

    # Goal: arbitrary logistics goal
    exps["small"] = update_dict(
        base,
        instances=[f'sample{i}.pddl' for i in [2]],
        # test_instances=["prob{:02d}.pddl".format(i) for i in range(2, 5)],
        test_instances=[],
        test_policy_instances=all_instances(),

        distance_feature_max_complexity=8,
        max_concept_size=8,

        use_equivalence_classes=True,
        # use_feature_dominance=True,
        use_incremental_refinement=True,
    )

    exps["debug"] = update_dict(
        exps["small"],

        instances=[f'sample{i}.pddl' for i in [2]],

        transition_classification_policy=debug_policy,
        # feature_generator=debug_features,
        use_incremental_refinement=False,
        use_equivalence_classes=True,
        use_feature_dominance=False,
    )

    return exps


def all_instances():
    return [f"prob0{i}.pddl" for i in range(1, 3)]


def goal_selector(lang):
    return "And(Equal(at_g,at),obj)"


def debug_features(lang):
    # undelivered packages:
    # And(Not(Equal(at_g,at)),obj)


    return [nwp, ready_to_rock, holding, nontable]


def debug_policy():
    wp = "And(And(Equal(on_g,on),Forall(Star(on),Equal(on_g,on))),Not(holding))"
    # wp = "And(Equal(on_g,on),Forall(Star(on),Equal(on_g,on)))"

    return [
        # Put down the held block on its target if possible
        [(holding, 'DEL'), (ready_to_rock, "DEL"), (nwp, 'INC')],

        # Put down the held block on table if cannot put it well placed
        [(holding, 'DEL'), (ready_to_rock, "=0"), (nontable, "INC")],
    ]
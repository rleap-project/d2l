
from sltp.util.misc import update_dict
from sltp.util.names import delivery_names


def experiments():
    base = dict(
        domain_dir="delivery",
        domain="domain.pddl",
        test_domain="domain.pddl",
        feature_namer=delivery_names,
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

    exps["small"] = update_dict(
        base,
        instances=[
            'instance_3_3_0.pddl',  # Use one small instance with three packages
            'instance_4_2_0.pddl',  # And a slightly larger one with two packages
            # 'instance_5_0.pddl',
        ],
        test_policy_instances=all_test_instances(),

        max_concept_size=8,
        distance_feature_max_complexity=14,

        # feature_generator=debug_features,
        # d2l_policy=debug_policy,

        use_equivalence_classes=True,
        # use_feature_dominance=True,
        # print_denotations=True,
    )

    return exps


def expected_features_wo_conditionals(lang):
    return [
        "Bool[And(loct,locp)]",
        "Bool[And(locp,Nominal(inside_taxi))]",
        "Bool[And(locp,locp_g)]",
        "Dist[loct;adjacent;locp]",
        "Dist[loct;adjacent;locp_g]",
    ]


def expected_features(lang):
    return [
        "Bool[And(locp,locp_g)]",  # Goal-distinguishing
        "Dist[loct;adjacent;locp]",  # Distance between taxi and passenger
        "Bool[And(loct,locp)]",
        "Bool[And(locp,Nominal(inside_taxi))]",
        "If{Bool[And(locp,Nominal(inside_taxi))]}{Dist[locp_g;adjacent;loct]}{Infty}",
    ]


def debug_features(lang):
    return [
        "Dist[Exists(Inverse(at),empty);adjacent;Exists(Inverse(at),And(Not(Equal(at_g,at)),package))]",
        "Dist[Exists(Inverse(at),truck);adjacent;Exists(Inverse(at_g),<universe>)]",

        "Bool[empty]",

        "Num[And(Not(Equal(at_g,at)),package)]",
    ]


def debug_policy():
    truck_empty = "Bool[empty]"
    dist_to_unpicked_package = "Dist[Exists(Inverse(at),empty);adjacent;Exists(Inverse(at),And(Not(Equal(at_g,at)),package))]"
    dist_to_target = "Dist[Exists(Inverse(at),truck);adjacent;Exists(Inverse(at_g),<universe>)]"
    undelivered = "Num[And(Not(Equal(at_g,at)),package)]"

    return [
        # If empty, move towards unpicked package if possible
        [(truck_empty, ">0"), (truck_empty, "NIL"), (dist_to_unpicked_package, 'DEC')],

        # If carrying something, move closer to target
        [(truck_empty, "=0"), (truck_empty, "NIL"), (dist_to_target, 'DEC')],

        # Picking up something is good as long as it's not a delivered package
        [(truck_empty, "DEL"), (undelivered, 'NIL')],

        # Leaving a package is good as long as it's on the target location
        [(truck_empty, "ADD"), (undelivered, 'DEC')],
    ]


def all_test_instances():
    instances = []
    for gridsize in [3, 4, 5, 7, 9]:
        for npacks in [2, 3]:
            for run in range(0, 3):
                instances.append(f"instance_{gridsize}_{npacks}_{run}.pddl")
    return instances

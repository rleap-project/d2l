from .misc import extend_namer_to_all_features


def gripper_names(feature):
    s = str(feature)
    base = {
        "Exists(at,Not(Nominal(roomb)))": "nballs-A",
        "Exists(at,Nominal(roomb))": "nballs-B",
        "Exists(carry,<universe>)": "ncarried",
        "And(at-robby,Nominal(roomb))": "robot-at-B",
        "Exists(at,Not(at-robby))": "nballs-in-rooms-with-no-robot",
        "free": "nfree-grippers",
        "Exists(at,Exists(Inverse(at-robby),<universe>))": "nballs-in-room-with-some-robot",
        "And(Exists(gripper,Exists(at-robby,{roomb})),free)": "nfree-grippers-at-B",
        "Exists(at-robby,{roomb})": "nrobots-at-B",
        "Exists(gripper,Exists(at-robby,{roomb}))": "ngrippers-at-B",
        "Exists(carry,Exists(gripper,Exists(at-robby,{roomb})))": "nballs-carried-in-B",
        "Exists(at,And(Forall(Inverse(at-robby),<empty>), Not({roomb})))":
            "nballs-in-some-room-notB-without-any-robot",
        "And(Exists(Inverse(at),<universe>), And({roomb}, Not(at-robby)))": "some-ball-in-B-but-robot-not-in-B",
        "And(Forall(Inverse(at),<empty>),room)": "num-empty-rooms",
        "Exists(at,And(at-robby,Nominal(roomb)))": "num-balls-at-B-when-robot-at-B-as-well",
        "Not(And(Forall(carry,<empty>),Forall(at,at-robby)))": "num-balls-either-carried-or-not-in-same-room-as-robot",
        # "Not(And(Not(And(at-robby,Nominal(roomb))),Forall(at,And(at-robby,Nominal(roomb)))))": "",
        # "Not(And(Not(And(Forall(at,at-robby),ball)),Not(And(at-robby,Nominal(roomb)))))": "",
        # "Not(And(Forall(at-robby,And(Not(Nominal(roomb)),Exists(Inverse(at),<universe>))),Forall(carry,<empty>)))":
        #     ""
        "And(Exists(carry,<universe>),Exists(at_g,at-robby))": "if-robot-at-B-then-num-carried-balls-else-emptyset",
    }
    return extend_namer_to_all_features(base).get(s, s)


def gripper_parameters(language):
    return [language.constant("roomb", "object")]


def spanner_names(feature):
    s = str(feature)
    base = {
        "And(tightened_g,Not(tightened))": "n-untightened-nuts",
        "Exists(carrying,<universe>)": "n-carried-spanners",
        "Forall(Inverse(link),<empty>)": "first-cell",  # Neat!
        "Exists(at,Forall(Inverse(link),<empty>))": "n-things-on-first-cell",
        "And(Exists(at,Exists(Inverse(at),man)),Not(man))": "n-spanners-in-same-cell-as-man",
        "And(Exists(at,Exists(Inverse(at),man)),spanner)":  "n-spanners-in-same-cell-as-man",
        "Exists(at,Exists(link,Exists(Inverse(at),<universe>)))": "",
        "loose": "n-untightened-nuts",
        "Exists(at,Exists(link,Exists(Inverse(at),man)))": "n-spanners-on-cell-left-to-man",
    }
    return extend_namer_to_all_features(base).get(s, s)
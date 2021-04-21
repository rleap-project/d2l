from .misc import extend_namer_to_all_features


def no_parameter(lang):
    return []


def gripper_names(feature):
    s = str(feature)
    base = {
        "Exists(at,Not(Nominal(roomb)))": "nballs-A",
        "Exists(at,Nominal(roomb))": "nballs-B",
        "Exists(carry,<universe>)": "ncarried",
        "And(at-robby,Nominal(roomb))": "robot-at-B",
        "Exists(at,at-robby)": "nballs-in-room-with-robot",
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
        "And(Exists(carry,<universe>),Exists(at_g,at-robby))": "if-robot-at-B-then-num-carried-balls-else-emptyset",
        "Exists(at_g,at-robby)": "robby-at-B",  # This one only works for a single target room
        "Not(Equal(at_g,at))": "n-balls-not-at-B",
    }
    return extend_namer_to_all_features(base).get(s, s)


def gripper_parameters(language):
    return [language.constant("roomb", "object")]


def spanner_names(feature):
    s = str(feature)
    base = {
        "And(tightened_g,Not(tightened))": "n-untightened-nuts",
        "Exists(Inverse(carrying),<universe>)": "n-carried-spanners",
        "Forall(Inverse(link),<empty>)": "first-cell",  # Neat!
        "Exists(at,Forall(Inverse(link),<empty>))": "n-things-on-first-cell",
        "And(Exists(at,Exists(Inverse(at),man)),Not(man))": "n-spanners-in-same-cell-as-man",
        "And(Exists(at,Exists(Inverse(at),man)),spanner)":  "n-spanners-in-same-cell-as-man",
        "Exists(at,Exists(link,Exists(Inverse(at),<universe>)))": "",
        "loose": "n-untightened-nuts",
        "Exists(at,Exists(link,Exists(Inverse(at),man)))": "n-spanners-on-cell-left-to-man",
        "Exists(Inverse(at),spanner)": "locations-with-a-spanner",
        "Exists(carrying,useable)": "bob-is-carrying-a-usable-spanner",
        "tightened": "n-tightened",
        "Exists(Star(link),Exists(Inverse(at),man))": "n-unreachable-locations",
        "Exists(at,Exists(Star(link),Exists(Inverse(at),man)))": "n-unreachable-spanners",
        "LessThan{Num[Exists(Inverse(carrying),<universe>)]}{Num[loose]}": "not-carrying-enough-spanners",
        "Exists(at,Forall(Inverse(at),man))": "bobs-loc-empty",
        "Exists(Star(link),Exists(Inverse(at),spanner))": "n-locs-from-which-some-spanner-is-reachable",
        "Exists(Inverse(Star(link)),Exists(Inverse(at),<universe>))": "n-locs-reachable-from-a-loc-with-things",
        "And(Forall(at,<empty>),useable)": "picked-up-spanners",

        "useable": "n-useable-spanners",
        "Exists(at,<universe>)": "n-items-somewhere",
        "Forall(at,<empty>)": "n-picked",
    }
    d = extend_namer_to_all_features(base)
    return d.get(s, s)


def blocksworld_names(feature):
    s = str(feature)
    base = {
        "And(clear,Nominal(a))": "clear(a)",
        "And(clear,Nominal(b))": "clear(b)",
        "holding": "holding(·)",
        "And(Nominal(a),holding)": "holding(a)",
        "And(holding,Nominal(a))": "holding(a)",
        "And(holding,Nominal(b))": "holding(b)",
        "And(Exists(on,Nominal(b)),Nominal(a))": "on(a,b)",
        "And(Exists(loc,Nominal(b)),Nominal(a))": "on(a,b)",  # FSTRIPS
        "And(Exists(Inverse(on),Nominal(a)),Nominal(b))": "on(a,b)",
        "And(Exists(Star(on),Nominal(b)),Nominal(a))": "above(a,b)",
        "And(Not(Nominal(a)),holding)": "H",
        "Exists(Inverse(on),Nominal(a))": "Z",
        "Exists(Star(on),Nominal(a))": "n(a)",
        "Exists(Star(on),Nominal(b))": "n(b)",
        "Exists(Star(loc),Nominal(a))": "n(a)",  # FSTRIPS
        "Exists(Star(loc),Nominal(b))": "n(b)",  # FSTRIPS
        "And(ontable,Nominal(a))": "ontable(a)",
        "And(Forall(on,Nominal(b)),Nominal(a))": "a_on_b_ontable_or_held",
        "And(And(And(Not(Exists(Star(on),Nominal(a))),Not(Exists(Star(Inverse(on)),Nominal(a)))),Not(Nominal(a))),Not(holding))": "m(a)",
        "And(And(Forall(Star(on),Not(Nominal(a))),Forall(Star(Inverse(on)),Not(Nominal(a)))),And(Not(holding),Not(Nominal(a))))": "m(a)",
        "Exists(Star(on),Exists(on,Nominal(b)))": "n-at-least-2-above-b",
        "Not(clear)": "n-unclear",
        "And(clear,ontable)": "n-single-blocks",
        "ontable": "n-ontable",
        "Atom[handempty]": "handempty",
        # "superficially-well-placed": all blocks below are the same as in goal
        "And(Equal(Star(on_g),Star(on)),clear)": "n-clear-and-superficially-well-placed-blocks",
        "Equal(Inverse(loc_g),Inverse(Star(loc)))": "n-blocks-below-their-hat",  # FSTRIPS
        "Exists(Star(loc),Exists(loc_g,Not(Equal(Inverse(loc_g),Inverse(loc)))))": "n-x-with-misplaced-block-below",  # FSTRIPS
        "clear": "n-clear",
        "And(Equal(loc_g,loc),Forall(Star(loc),Equal(loc_g,loc)))": "n-well-placed",  # FSTRIPS
        "And(Forall(Star(loc),Equal(loc_g,loc)),Equal(loc_g,loc))": "n-well-placed",  # Same as above, just rearranging AND elements
        "And(Equal(on_g,on),Forall(Star(on),Equal(on_g,on)))": "n-well-placed",
        "And(Forall(Star(on),Equal(on_g,on)),Equal(on_g,on))": "n-well-placed",
        "Equal(Star(loc_g),Star(loc))": "n-superficially-well-placed",  # FSTRIPS
        "Equal(Star(on_g),Star(on))": "n-superficially-well-placed",
        "Not(Equal(Star(on_g),Star(on)))": "n-superficially-misplaced",
        "Equal(loc_g,loc)": "n-ontarget",  # FSTRIPS
        "Not(Equal(loc_g,loc))": "n-not-ontarget",  # FSTRIPS
        "Equal(on_g,on)": "n-ontarget",
        "Not(Equal(on_g,on))": "n-not-ontarget",
        "Equal(Inverse(loc_g),Inverse(loc))": "n-right-under-target",  # FSTRIPS
        "Forall(Star(loc),Equal(loc_g,loc))": "n-all-below-well-placed",  # FSTRIPS
        "Forall(Star(on),Equal(on_g,on))": "n-all-below-well-placed",
        "Exists(on,Nominal(b))": "something-on(b)",
        "Exists(on,Nominal(a))": "something-on(a)",


        "Not(And(Equal(on_g,on),Equal(Star(on_g),Star(on))))": "not-ontarget-or-not-sup-well-placed",

        "Exists(Star(on),Not(Equal(on_g,on)))": "some_below_misplaced",
        "And(And(Exists(on_g,clear),Forall(Star(on),Equal(on_g,on))),holding)": "ready_to_rock",
    }

    return extend_namer_to_all_features(base).get(s, s)


def blocksworld_parameters_for_clear(language):
    # We simply add block "a" as a domain constant
    return [language.constant("a", "object")]


def blocksworld_parameters_for_on(language):
    return [language.constant("a", "object"), language.constant("b", "object")]


def blocksworld_declared_parameters_for_clear(language):
    # We simply add block "a" as a domain constant
    return [language.get("a")]


def blocksworld_declared_parameters_for_on(language):
    return [language.get("a"), language.get("b")]


def reward_names(feature):
    s = str(feature)
    base = {
        "reward": "n-rewards",
        "Dist[at;adjacent;reward]": "dist-to-closest-reward",
        "Dist[at;Restrict(adjacent,unblocked);reward]": "unblocked-dist-to-closest-reward",
    }

    return extend_namer_to_all_features(base).get(s, s)


def hanoi_names(feature):
    s = str(feature)
    base = {
        "Equal(on_g,on)": "n-ontarget",
        "Exists(Star(on),Not(Equal(on_g,on)))": "n-has-misplaced-disc-below",
        "Forall(on_g,clear)": "n-target-is-clear",
    }

    return extend_namer_to_all_features(base).get(s, s)


def logistics_names(feature):
    s = str(feature)
    base = {
        "Exists(in,<universe>)": "n_loaded_packages",  # both in plane or truck
        "And(Not(Equal(at_g,at)),obj)": "n_undelivered_packages",
        "Exists(at_g,Exists(Inverse(at),airplane))": "num-packages-whose-destiny-has-an-airplane",
        "Exists(at_g,Exists(Inverse(at),<universe>))": "num-packages-with-destiny",
        "Exists(at_g,airport)": "num-packages-whose-destiny-has-an-airport",
        "Exists(in,airplane)": "num-packages-in-airplane",
        # "And(Equal(at_g,Inverse(at)),airport)": "",
        "And(Exists(at,airport),obj)": "num-packages-at-city-with-airport",
        # "Exists(at,Forall(Inverse(at),truck))": "",
    }

    return extend_namer_to_all_features(base).get(s, s)


def gridworld_names(feature):
    s = str(feature)
    base = {
        "And(goal_xpos,xpos)": "on-x-target",
        "And(goal_ypos,ypos)": "on-y-target",
        "And(xpos,ypos)": "on-diagonal",
        "Exists(succ,ypos)": "not-on-left-limit",
        "Exists(Star(succ),xpos)": "xpos",
    }

    return extend_namer_to_all_features(base).get(s, s)


def miconic_names(feature):
    s = str(feature)
    base = {
        "And(lift-at,Exists(Inverse(destin),boarded))": "lift_at_dest_some_boarded_pass",
        "served": "n-served",
        "boarded": "n-boarded",

        # num passengers unboarded s.t. lift is on their origin floor
        "And(And(Not(boarded),Not(served)),Exists(origin,lift-at))": "n_pass_ready_to_board",
        "And(lift-at,Exists(Inverse(origin),And(Not(boarded),Not(served))))": "lift_at_origin_some_awaiting_pass",

        "Exists(origin,lift-at)": "n-pass-waiting-on-lifts-floor",
        "And(Exists(destin,lift-at),boarded)": "n-boarded-and-at-destiny",
        "Forall(origin,lift-at)": "n-ready-to-board",

    }

    return extend_namer_to_all_features(base).get(s, s)


def satellite_names(feature):
    s = str(feature)
    base = {
        "calibrated": "n-calibrated-instruments",
        "power_on": "n-on-instruments",
        "": "",
        "": "",
    }
    return extend_namer_to_all_features(base).get(s, s)


def delivery_names(feature):
    s = str(feature)
    base = {
        "And(locp,locp_g)": "passenger-delivered",
        "And(loct,locp)": "at-passenger-location",
        "And(locp,Nominal(inside_taxi))": "passenger-in-taxi",
        "Dist[loct;adjacent;locp]": "dist-to-passenger",
        "Dist[locp;adjacent;loct]": "dist-to-passenger",
        "Dist[loct;Restrict(adjacent,<universe>);locp]": "dist-to-passenger",
        "Dist[locp_g;adjacent;loct]": "dist-to-passenger-target",
        "If{Bool[And(locp,Nominal(inside_taxi))]}{Dist[locp_g;adjacent;loct]}{Infty}": "cond-dist-to-dest",

        "Dist[Exists(Inverse(at),empty);adjacent;Exists(Inverse(at),And(Not(Equal(at_g,at)),package))]": "dist-to-undelivered",

        "empty": "empty",
        "And(Not(Equal(at_g,at)),package)": "n-undelivered",

        # This one encodes the distance between a truck and any cell with a package that is not the target cell
        "Dist[Exists(Inverse(at),truck);adjacent;And(Forall(Inverse(at_g),<empty>),Exists(Inverse(at),package))]": "dist-to-cell-with-undelivered-p",


        "Dist[Exists(Inverse(at_g),<universe>);adjacent;Exists(Inverse(at),truck)]": "dist-to-target",
        "Dist[Exists(Inverse(at),truck);adjacent;Exists(Inverse(at_g),<universe>)]": "dist-to-target",
    }

    return extend_namer_to_all_features(base).get(s, s)


def visitall_names(feature):
    s = str(feature)
    base = {
        "Dist[at-robot;connected;Not(visited)]": "dist-to-closest-unvisited",
        "Dist[at-robot;connected;Exists(connected,Not(visited))]": "dist-to-closest-adj-to-unvisited",
        "visited": "n-visited",
        "Not(visited)": "n-unvisited",
    }
    return extend_namer_to_all_features(base).get(s, s)


def childsnack_names(feature):
    s = str(feature)
    base = {
        # "": "",
        "served": "num-served-children",
        "And(Not(served),child)": "num-unserved-children",
        "notexist": "num-unprepared-sandwiches",
        "no_gluten_sandwich": "num-sandwiches-wo-gluten",
        "at_kitchen_bread": "num-breads-at-kitchen",
        "at_kitchen_content": "num-fillings-at-kitchen",
        "at_kitchen_sandwich": "num-sandwiches-at-kitchen",
        "Exists(ontray,<universe>)": "num-sandwiches-on-some-tray",
        "And(allergic_gluten,served)": "num-allergic-served",
        "Exists(at,Nominal(kitchen))": "num-trays-on-kitchen",
        "And(Not(served),not_allergic_gluten)": "num-unallergic-unserved",
        "And(Not(served),allergic_gluten)": "num-allergic-unserved",
        "And(Not(no_gluten_content),content-portion)": "num-gluten-free-fillings",
        "And(Not(no_gluten_bread),bread-portion)": "num-gluten-free-breads",
        "And(Not(no_gluten_sandwich),sandwich)": "num-sandwiches-with-gluten",
        "Exists(at,Exists(Inverse(waiting),<universe>))": "num-trays-on-place-with-some-child",
        "Exists(ontray,Exists(at,Nominal(kitchen)))": "num-sandwiches-on-some-tray-in-kitchen",
    }

    return extend_namer_to_all_features(base).get(s, s)


def floortile_names(feature):
    s = str(feature)
    base = {
        "Exists(at,Not(Nominal(roomb)))": "nballs-A",
        "Exists(at,Nominal(roomb))": "nballs-B",
        "Exists(carry,<universe>)": "ncarried",
    }
    return extend_namer_to_all_features(base).get(s, s)


def barman_names(feature):
    s = str(feature)
    base = {
        # "": "",
    }

    return extend_namer_to_all_features(base).get(s, s)

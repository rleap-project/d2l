#! /usr/bin/env python3
import os
import sys
from pathlib import Path

from tarski.io import FstripsReader
from tarski.model import unwrap_tuple

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))


def main(maintenance_instance):
    inst = Path(maintenance_instance)
    reader = FstripsReader(raise_on_error=True)
    problem = reader.read_problem(inst.parent / "domain.pddl", inst)
    lang = problem.language
    schedule = [unwrap_tuple(tup) for tup in problem.init.get_extension(lang.get('at'))]

    visitobjs = [f'v{i}' for i in range(1, len(schedule)+1)]
    print(f'  {" ".join(visitobjs)} - visit\n\n')
    for i, (plane, day, airport) in enumerate(schedule, start=1):
        vname = f'v{i}'
        print(f'    ;; (at {plane} {day} {airport})')
        print(f'    (which {vname} {plane}) (where {vname} {airport}) (when_ {vname} {day})\n')


if __name__ == "__main__":
    main(sys.argv[1])

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os

from tarski.benchmarks.blocksworld import generate_random_bw_pattern
from tarski.fstrips import create_fstrips_problem, fstrips
from tarski.io import FstripsWriter
from tarski.syntax import land
from tarski import fstrips as fs

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))


def generate_problem(nblocks, run):
    lang = generate_atomic_bw_language(nblocks)
    problem = create_fstrips_problem(lang,
                                     domain_name="blocksworld-atomic",
                                     problem_name=f'instance-{nblocks}-{run}')

    clear, on, diff, table = lang.get('clear', 'on', 'diff', 'table')

    # Generate init pattern
    clearplaces, locations = generate_random_bw_pattern(lang)
    for x, y in locations:
        problem.init.add(on, lang.get(x), lang.get(y))
    for x in clearplaces:
        problem.init.add(clear, lang.get(x))

    # Add the quadratic number of (static) diff(b, c) atoms
    for x, y in itertools.permutations(lang.constants(), 2):
        problem.init.add(diff, x, y)

    # Generate goal pattern
    _, locations = generate_random_bw_pattern(lang)
    conjuncts = []
    for x, y in locations:
        conjuncts.append(on(lang.get(x), lang.get(y)))
    problem.goal = land(*conjuncts, flat=True)

    b, x, y = [lang.variable(name, 'object') for name in ['b', 'x', 'y']]

    problem.action('move', [b, x, y],
                   precondition=land(diff(b, table), diff(y, table), diff(b, y), clear(b), on(b, x), clear(y),
                                     flat=True),
                   effects=[fs.DelEffect(on(b, x)),
                            fs.AddEffect(clear(x)),
                            fs.DelEffect(clear(y)),
                            fs.AddEffect(on(b, y))])

    problem.action('move-to-table', [b, x],
                   precondition=land(diff(b, table), diff(x, table), clear(b), on(b, x),
                                     flat=True),
                   effects=[fs.DelEffect(on(b, x)),
                            fs.AddEffect(clear(x)),
                            fs.AddEffect(on(b, table))])

    return problem, [table]


def generate_atomic_bw_language(nblocks):
    """ A STRIPS encoding of BW with a single move action. """
    lang = fstrips.language("blocksworld-atomic")

    lang.predicate('clear', 'object')
    lang.predicate('on', 'object', 'object')
    lang.predicate('diff', 'object', 'object')

    lang.constant('table', 'object')
    _ = [lang.constant(f'b{k}', 'object') for k in range(1, nblocks+1)]

    return lang


def main():
    for nblocks in range(6, 31):
        for run in range(0, 5):
            problem, domain_constants = generate_problem(nblocks=nblocks, run=run)
            writer = FstripsWriter(problem)
            writer.write(domain_filename=os.path.join(_CURRENT_DIR_, "domain_atomic.pddl"),
                         instance_filename=os.path.join(_CURRENT_DIR_, f"test_atomic_{nblocks}_{run}.pddl"),
                         domain_constants=domain_constants)


if __name__ == "__main__":
    main()

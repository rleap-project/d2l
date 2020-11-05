
from tarski.io import FstripsReader


def parse_pddl(domain_file, instance_file=None):
    """ Parse the given PDDL domain and instance files, the latter only if provided """
    reader = FstripsReader()
    reader.parse_domain(domain_file)
    problem = reader.problem

    # The generic constants are those which are parsed before parsing the instance file
    generic_constants = problem.language.constants()

    if instance_file is not None:
        reader.parse_instance(instance_file)

    return problem, problem.language, generic_constants

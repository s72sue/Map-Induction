import os
import sys
from collections import namedtuple

Domain = namedtuple('Domain', ['name', 'requirements', 'types', 'type_dict', 'constants',
                               'predicates', 'predicate_dict', 'functions', 'actions', 'axioms'])

Problem = namedtuple('Problem', ['task_name', 'task_domain_name', 'task_requirements',
                                 'objects', 'init', 'goal', 'use_metric'])

# Fast downward translation requires command line arguments

def read(filename):
    with open(filename, 'r') as f:
        return f.read()

        
def find_build(fd_path):
    for release in ['release']:  # TODO: list the directory
        path = os.path.join(fd_path, 'builds/{}/'.format(release))
        if os.path.exists(path):
            return path
    # TODO: could also just automatically compile
    raise RuntimeError('Please compile FastDownward first')


FD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../downward/')
print(FD_PATH)
TRANSLATE_PATH = os.path.join(find_build(FD_PATH), 'bin/translate')
DOMAIN_INPUT = 'domain.pddl'
PROBLEM_INPUT = 'problem.pddl'
TRANSLATE_FLAGS = []
original_argv = sys.argv[:]
sys.argv = sys.argv[:1] + TRANSLATE_FLAGS + [DOMAIN_INPUT, PROBLEM_INPUT]
sys.path.append(TRANSLATE_PATH)

from downward.src.translate.pddl_parser.parsing_functions import parse_domain_pddl, parse_task_pddl, \
    parse_condition, check_for_duplicates
import pddl_parser.lisp_parser
import pddl_parser
import subprocess
import os
import re
from collections import namedtuple
import instantiate
import normalize
import copy
import pddl
import build_model
import pddl_to_prolog
from downward.src.translate.pddl_parser.parsing_functions import parse_domain_pddl, parse_task_pddl, \
    parse_condition, check_for_duplicates

sys.argv = original_argv


def parse_lisp(lisp):
    return pddl_parser.lisp_parser.parse_nested_list(lisp.splitlines())


def parse_problem(domain, problem_pddl):
    return Problem(*parse_task_pddl(parse_lisp(problem_pddl), domain.type_dict, domain.predicate_dict))


def parse_sequential_domain(domain_pddl):
    domain = Domain(*parse_domain_pddl(parse_lisp(domain_pddl)))
    return domain


if __name__ == '__main__':
    print("Testing pddl parsing ...")
    test_problem_file = "./task_planning_problems/tasks/gridYx/Grid10x.pddl"
    test_domain_file = "./task_planning_problems/tasks/gridYx/GridYx_domain.pddl"
    domain = parse_sequential_domain(read(test_domain_file))
    problem = parse_problem(domain, read(test_problem_file))
    print(domain, problem)

DEFAULT_MAX_TIME = 30  # INF
DEFAULT_PLANNER = 'ff-astar'
FD_PATH = "./downward"
OBJECT = 'object'
Problem = namedtuple('Problem', ['task_name', 'task_domain_name', 'task_requirements',
                                 'objects', 'init', 'goal', 'use_metric'])


class MockSet(object):
    def __init__(self, test=lambda item: True):
        self.test = test

    def __contains__(self, item):
        return self.test(item)


def objects_from_evaluations(evaluations):
    # TODO: assumes object predicates
    objects = set()
    for evaluation in evaluations:
        objects.update(set(list(evaluation[1:])))

    return objects


def fd_from_evaluation(evaluation):
    name = evaluation[0]
    args = evaluation[1:]
    fluent = pddl.f_expression.PrimitiveNumericExpression(symbol=name, args=args)
    expression = pddl.f_expression.NumericConstant(True)
    return pddl.f_expression.Assign(fluent, expression)


def get_plan_filename(pf_index):
    return "./plan." + str(pf_index)


def plan(domain_file, problem_file):
    FD_BIN = "./downward/fast-downward.py"
    commandline_args = "--plan-file plan --alias seq-sat-lama-2011"
    # commandline_args = ""
    # other_args = "--evaluator \"hff=ff()\" --evaluator \"hcea=cea()\" --search \"lazy_greedy([hff, hcea], preferred=[hff, hcea])\""
    # other_args = "--search \"astar(lmcut())\""
    other_args = ""
    command = "%s %s %s %s %s" % (FD_BIN, commandline_args, domain_file, problem_file, other_args)
    print(command)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, cwd=None, close_fds=True)
    try:
        output, error = proc.communicate(timeout=10)
        print("Proc Error: " + str(error))
    except:
        print("Timeout")
    pf_index = 1
    # Need to get the latest plan file
    while (os.path.isfile(get_plan_filename(pf_index))):
        pf_index += 1

    plan_file = get_plan_filename(pf_index - 1)
    return plan_file


def find_plan(domain_file, problem_file):
    plan_file = plan(domain_file, problem_file)
    return parse_solution(plan_file)


##################################################
Action = namedtuple('Action', ['name', 'args'])


def parse_action(line):
    entries = line.strip('( )').split(' ')
    name = entries[0]
    args = tuple(entries[1:])
    return Action(name, args)


def parse_solution(solution_file):
    with open(solution_file, 'r') as f:
        solution = f.read()

    # action_regex = r'\((\w+(\s+\w+)\)' # TODO: regex
    if solution is None:
        return None, cost
    cost_regex = r'cost\s*=\s*(\d+)'
    matches = re.findall(cost_regex, solution)
    # TODO: recover the actual cost of the plan from the evaluations
    lines = solution.split('\n')[:-2]  # Last line is newline, second to last is cost
    plan = list(map(parse_action, lines))
    return plan


def literal_holds(state, literal):
    # return (literal in state) != literal.negated
    return (literal.positive() in state) != literal.negated


def conditions_hold(state, conditions):
    return all(literal_holds(state, cond) for cond in conditions)


def apply_action(state, action):
    state = set(state)
    assert (isinstance(action, pddl.PropositionalAction))
    # TODO: signed literals
    for conditions, effect in action.del_effects:
        if conditions_hold(state, conditions):
            state.discard(effect)
    for conditions, effect in action.add_effects:
        if conditions_hold(state, conditions):
            state.add(effect)
    return state


def task_from_domain_problem(domain, problem):
    task_name, task_domain_name, task_requirements, objects, init, goal, use_metric = problem

    assert domain.name == task_domain_name
    requirements = pddl.Requirements(sorted(set(domain.requirements.requirements +
                                                task_requirements.requirements)))
    objects = domain.constants + objects
    init.extend(pddl.Atom("=", (obj.name, obj.name)) for obj in objects)

    task = pddl.Task(domain.name, task_name, requirements, domain.types, objects,
                     domain.predicates, domain.functions, init, goal,
                     domain.actions, domain.axioms, use_metric)
    normalize.normalize(task)
    # task.add_axiom
    return task


def find_unique(test, sequence):
    found, value = False, None
    for item in sequence:
        if test(item):
            if found:
                raise RuntimeError('Both elements {} and {} satisfy the test'.format(value, item))
            found, value = True, item
    if not found:
        raise RuntimeError('Unable to find an element satisfying the test')
    return value


def pddl_from_object(obj):
    # if isinstance(obj, str):
    #   return obj
    # return obj.pddl
    return str(obj)


def get_function_assignments(task):
    return {f.fluent: f.expression for f in task.init
            if isinstance(f, pddl.f_expression.FunctionAssignment)}


def t_get_action_instances(task, action_plan):
    type_to_objects = instantiate.get_objects_by_type(task.objects, task.types)
    function_assignments = get_function_assignments(task)
    fluent_facts = MockSet()
    init_facts = set()
    action_instances = []
    predicate_to_atoms = {}

    for name, objects in action_plan:
        # TODO: what if more than one action of the same name due to normalization?
        # Normalized actions have same effects, so I just have to pick one
        action = find_unique(lambda a: a.name == name, task.actions)
        args = list(map(pddl_from_object, objects))
        assert (len(action.parameters) == len(args))
        variable_mapping = {p.name: a for p, a in zip(action.parameters, args)}
        instance = action.instantiate(variable_mapping, init_facts,
                                      fluent_facts, type_to_objects,
                                      task.use_min_cost_metric, function_assignments, predicate_to_atoms)
        assert (instance is not None)
        action_instances.append(instance)
    return action_instances


def get_action_instances(task, action_plan):
    type_to_objects = instantiate.get_objects_by_type(task.objects, task.types)
    function_assignments = get_function_assignments(task)
    model = build_model.compute_model(pddl_to_prolog.translate(task))
    fluent_facts = instantiate.get_fluent_facts(task, model)
    init_facts = task.init
    action_instances = []
    for name, objects in action_plan:
        # TODO: what if more than one action of the same name due to normalization?
        # Normalized actions have same effects, so I just have to pick one
        action = find_unique(lambda a: a.name == name, task.actions)

        args = list(map(pddl_from_object, objects))
        assert (len(action.parameters) == len(args))
        variable_mapping = {p.name: a for p, a in zip(action.parameters, args)}
        instance = action.instantiate(variable_mapping, init_facts, function_assignments,
                                      fluent_facts, type_to_objects,
                                      task.use_min_cost_metric)
        assert (instance is not None)
        action_instances.append(instance)
    return action_instances

import tsplib95 as tsplib
import tempfile
import subprocess
import shutil


def solve(solver='LKH', problem=None, **params):
    assert shutil.which(solver) is not None, f'{solver} not found.'

    valid_problem = problem is not None and isinstance(problem, tsplib.models.StandardProblem)
    assert ('problem_file' in params) ^ valid_problem, 'Specify a TSPLIB95 problem object *or* a path.'
    if problem is not None:
        # hack for bug in tsplib
        if len(problem.depots) > 0:
            problem.depots = map(lambda x: f'{x}\n', problem.depots)

        prob_file = tempfile.NamedTemporaryFile(mode='w')
        problem.write(prob_file)
        prob_file.write('\n')
        prob_file.flush()
        params['problem_file'] = prob_file.name

    if 'tour_file' not in params:
        tour_file = tempfile.NamedTemporaryFile(mode='w')
        params['tour_file'] = tour_file.name

    with tempfile.NamedTemporaryFile(mode='w') as par_file:
        par_file.write('SPECIAL\n')
        for k, v in params.items():
            par_file.write(f'{k.upper()} = {v}\n')
        par_file.flush()

        try:
            subprocess.check_output([solver, par_file.name], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise Exception(e.output.decode())

        solution = tsplib.load(params['tour_file'])
        return solution.tours

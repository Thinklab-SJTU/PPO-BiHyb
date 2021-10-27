import torch
import random
import os
import glob
import re
from copy import deepcopy
import time
import numpy as np
import tsplib95
from tsp_algorithms import calc_furthest_insertion_tour_len, calc_lkh_tour_len, calc_nearest_neighbor_tour_len,\
    get_lower_matrix, solveFarthestInsertion
from tsp_main import parse_tsp
from utils import random_triangulate

VERY_LARGE_INT = 10 # 65536


class TSPEnv(object):
    def __init__(self, solver_type='nn', min_size=100, max_size=200):
        self.solver_type = solver_type
        self.min_size = min_size
        self.max_size = max_size
        self.process_dataset()
        self.available_solvers = ('nn','furthest', 'lkh-fast','lkh-accu')
        assert solver_type in self.available_solvers

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")
        dirpath = './hcp_data'
        self.tspfiles = []

        def get_size(elem):
            prob_size = re.findall('[0-9]+', elem)
            if len(prob_size) != 1:
                prob_size = prob_size[1]
            else:
                prob_size = prob_size[0]
            return int(prob_size)

        for fp in glob.iglob(os.path.join(dirpath, "*.hcp")):
            if self.min_size <= get_size(fp) <= self.max_size:
                self.tspfiles.append(fp)
        self.tspfiles.sort(key=get_size)

    def generate_tuples(self, num_train_samples, num_test_samples, rand_id):
        random.seed(int(rand_id))
        np.random.seed(int(rand_id + num_train_samples))
        assert num_train_samples + num_test_samples <= len(self.tspfiles), \
            f'{num_train_samples + num_test_samples} > {len(self.tspfiles)}'
        #train_test_list = random.sample(self.tspfiles, num_train_samples + num_test_samples)

        training_tuples = []
        testing_tuples = []

        return_tuples = training_tuples
        sum_num_nodes = 0
        for i, tsp_path in enumerate(self.tspfiles):
            '''
            ####################################################
            #     generate problems by random triangulation    #
            ####################################################
            class Prob:
                dimension = 0
                name = ''
            problem = Prob()
            problem.dimension = 500
            problem.name = 'hcp'
            new_adj = random_triangulate(problem.dimension)
            
            lower_left_matrix = []
            for r in range(problem.dimension):
                lower_left_matrix.append([])
                for c in range(r+1):
                    lower_left_matrix[r].append(0)

            for r, line in enumerate(lower_left_matrix):
                for c, weight in enumerate(line):
                    if new_adj[r, c] == 0: # or random.random() < 0.1:
                        lower_left_matrix[r][c] = VERY_LARGE_INT  #random.randint(0, 1000)
                    else:
                        lower_left_matrix[r][c] = 1
            '''

            ####################################################
            #        load real-world TSP (HCP) instances       #
            ####################################################
            problem = tsplib95.load(tsp_path)
            lower_left_matrix = get_lower_matrix(problem, 1, 2)

            tsp_solutions = {}
            tsp_times = {}
            for key in self.available_solvers:
                tour, sol, sec = self.solve_feasible_tsp(lower_left_matrix, key)
                tsp_solutions[key] = sol
                tsp_times[key] = sec
                if key == self.solver_type:
                    edge_candidates = self.edge_candidate_from_tour(tour, problem.dimension)
            print(f'id {i} {problem.name} {problem.dimension} '
                  f'{"; ".join([f"{x} tour={tsp_solutions[x]:.2f} time={tsp_times[x]:.2f}" for x in self.available_solvers])}')
            sum_num_nodes += problem.dimension
            return_tuples.append((
                lower_left_matrix,  # lower-left triangle of adjacency matrix
                edge_candidates,  # edge candidates
                tsp_solutions[self.solver_type],  # reference TSP solution
                tsp_solutions,  # all TSP solutions
                tsp_times,  # TSP solving time
            ))
            if i == num_train_samples - 1 or i == num_train_samples + num_test_samples - 1:
                print(f'average number of nodes: {sum_num_nodes / len(return_tuples)}')
                sum_num_nodes = 0
                for solver_name in self.available_solvers:
                    print(f'{solver_name} tour_len '
                          f'mean={np.mean([tup[3][solver_name] for tup in return_tuples], dtype=np.float):.4f} '
                          f'std={np.std([tup[3][solver_name] for tup in return_tuples], dtype=np.float):.4f}')
                return_tuples = testing_tuples
                if len(testing_tuples) > 0:
                    break
        return training_tuples, testing_tuples

    def step(self, list_lower_matrix, act, prev_solution):
        new_list_lower_matrix = deepcopy(list_lower_matrix)
        if isinstance(act, torch.Tensor):
            act = (act[0].item(), act[1].item())
        if act[0] >= act[1]:
            idx0, idx1 = act[0], act[1]
        else:
            idx0, idx1 = act[1], act[0]
        new_list_lower_matrix[idx0][idx1] += VERY_LARGE_INT
        new_tour, new_solution, _ = self.solve_feasible_tsp(new_list_lower_matrix, self.solver_type)
        new_edge_candidate = self.edge_candidate_from_tour(new_tour, len(new_list_lower_matrix))
        reward = prev_solution - new_solution
        done = new_solution == 0
        #done = False
        return reward, new_list_lower_matrix, new_edge_candidate, new_solution, done

    def step_e2e(self, list_lower_matrix, prev_act, act, prev_solution):
        new_list_lower_matrix = deepcopy(list_lower_matrix)
        if isinstance(prev_act, torch.Tensor):
            prev_act = prev_act.item()
        if isinstance(act, torch.Tensor):
            act = act.item()
        if prev_act is not None:
            if prev_act > act:
                new_list_lower_matrix[prev_act][act] += VERY_LARGE_INT
                step_cost = list_lower_matrix[prev_act][act]
            else:
                new_list_lower_matrix[act][prev_act] += VERY_LARGE_INT
                step_cost = list_lower_matrix[act][prev_act]
        else:
            step_cost = 0
        new_solution = prev_solution + step_cost
        node_candidates = self.get_node_candidates(new_list_lower_matrix, len(new_list_lower_matrix))
        if len(node_candidates) == 0:
            done = True
            last_act1, last_act2 = self.get_node_candidates(new_list_lower_matrix, len(new_list_lower_matrix), last_act=True)
            if last_act1 > last_act2:
                last_step_cost = list_lower_matrix[last_act1][last_act2]
            else:
                last_step_cost = list_lower_matrix[last_act2][last_act1]
            new_solution = new_solution + last_step_cost
        else:
            done = False
        return prev_solution - new_solution, new_list_lower_matrix, node_candidates, new_solution, done

    def solve_feasible_tsp(self, lower_left_matrix, solver_type):
        prev_time = time.time()
        tsp_inst = tsplib95.parse(parse_tsp(lower_left_matrix))
        if solver_type == 'nn':
            tour, length = calc_nearest_neighbor_tour_len(tsp_inst)
        elif solver_type == 'furthest':
            tour, length = solveFarthestInsertion(tsp_inst)
        elif solver_type == 'lkh-fast':
            tour, length = calc_lkh_tour_len(tsp_inst, move_type=5, runs=5)
        elif solver_type == 'lkh-accu':
            tour, length = calc_lkh_tour_len(tsp_inst, move_type=5, runs=10, max_trials=10000)
        else:
            raise ValueError(f'{solver_type} is not implemented.')
        comp_time = time.time() - prev_time
        return tour, length - tsp_inst.dimension, comp_time

    @staticmethod
    def edge_candidate_from_tour(tour, num_nodes):
        assert tour[0] == tour[-1]
        edge_candidate = {x: set() for x in range(num_nodes)}
        iter_obj = iter(tour)
        last_node = next(iter_obj)
        for node in iter_obj:
            edge_candidate[last_node].add(node)
            edge_candidate[node].add(last_node)
            last_node = node
        return edge_candidate

    @staticmethod
    def get_node_candidates(list_lower_mat, num_nodes, last_act=False):
        visited_once = set()
        visited_twice = set()
        for i in range(num_nodes):
            for j in range(i+1):
                if i != j:
                    if list_lower_mat[i][j] > VERY_LARGE_INT:
                        if i in visited_once:
                            visited_twice.add(i)
                            visited_once.remove(i)
                        else:
                            visited_once.add(i)
                        if j in visited_once:
                            visited_twice.add(j)
                            visited_once.remove(j)
                        else:
                            visited_once.add(j)
        if last_act:
            assert len(visited_once) == 2
            return visited_once
        else:
            candidates = list(range(num_nodes))
            for i in visited_twice:
                candidates.remove(i)
            for i in visited_once:
                candidates.remove(i)
            return candidates

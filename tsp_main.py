#!/usr/bin/env python3

from tsp_argparser  import parser
from tsp_algorithms import (calc_nearest_neighbor_tour
                        , calc_nearest_neighbor_tour_len
                        , calc_lkh_tour_len
                        , calc_furthest_insertion_tour
                        , calc_lkh_tour
                        , calc_furthest_insertion_tour_len
                        , get_lower_matrix
                        , get_edge_dict)
# from pyconcorde.concorde.tsp import TSPSolver

import time
import sys
from glob    import iglob
from os.path import isfile, isdir, join, exists
import numpy as np
import requests
import tsplib95
import random

sys.setrecursionlimit(10000)

def get_tsp_files(path_arg_list):
    for path_arg in path_arg_list:

        if isdir(path_arg):
            for filepath in iglob(join(path_arg,"*.tsp")):
                yield filepath

        elif isfile(path_arg) & str(path_arg).endswith(".tsp"):
            yield path_arg

        elif isfile(path_arg) & (not path_arg.endswith(".tsp")):
            print("Can't open file ``{0}'': not a .tsp file".format(path_arg))

        elif exists(path_arg):
            print("Path {0} is neither a file nor a directory".format(path_arg))

        else:
            print("Path {0} does not exist".format(path_arg))


def parse_tsp(list_m, dim=None, name='unknown'):
    if dim is None:
        dim = len(list_m)
    outstr = ''
    outstr += 'NAME: %s\n' % name #problem.name
    outstr += 'TYPE: TSP\n'
    outstr += 'COMMENT: %s\n' % name
    outstr += 'DIMENSION: %d\n' % dim #problem.dimension
    outstr += 'EDGE_WEIGHT_TYPE: EXPLICIT\n'
    outstr += 'EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW\n'
    outstr += 'EDGE_WEIGHT_SECTION:\n'
    for l in list_m:
        listToStr = ' '.join([str(elem) for elem in l])
        outstr += ' %s\n' % listToStr
    #outstr += 'EDGE_DATA_FORMAT: EDGE_LIST\n'
    #outstr += 'EDGE_DATA_SECTION:\n'
    #for edge_idx, weight in edges_dict.items():
    #    outstr += f' {edge_idx[0]+1} {edge_idx[1]+1} {weight}\n'
    #outstr += '-1\n'

    return outstr


def print_results_from_tsp_path(call_args, tsp_path):
    t_s = time.perf_counter()

    # load .tsp file (NODE_COORD)
    problem = tsplib95.load(tsp_path)
    print("TSP Problem:              {}".format(problem.name))

    # get Lower triangular matrix
    list_lower_matrix = get_lower_matrix(problem)
    # load the new tsp
    tsp = tsplib95.parse(parse_tsp(list_lower_matrix, problem.dimension, problem.name))
    
    if call_args.need_lkh:
        print("LKH TOUR LENGTH:     {}"
             . format(calc_lkh_tour_len(tsp)[1]))

    if call_args.need_nearest_neighbor:
        print("NEAREST NEIGHBOR LENGTH:  {}"
             . format(calc_nearest_neighbor_tour_len(tsp)[1]))

    if call_args.need_furthest_neighbor:
        print("FURTHEST INSERTION LENGTH: {}"
             . format(calc_furthest_insertion_tour_len(tsp)[1]))
    t_e = time.perf_counter()
    
    print("time: ",t_e-t_s)
    print("")
    del(tsp)


def main():
    call_args = parser.parse_args()
    for tsp_path in get_tsp_files(call_args.tsp_queue):
        print_results_from_tsp_path(call_args,tsp_path)


if __name__ == "__main__":
    main()

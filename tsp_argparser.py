import argparse

parser = argparse.ArgumentParser(
      description = "Parse TSP files and calculate paths using heuristic algorithms.")

parser.add_argument (
      "-n"
    , "--nearest"
    , action  = "store_true"
    , dest    = "need_nearest_neighbor"
    , default = False
    , help    = "calculate distance traveled by nearest neighbor"
    )

parser.add_argument (
      "-f"
    , "--furthest"
    , action  = "store_true"
    , dest    = "need_furthest_neighbor"
    , default = False
    , help    = "calculate distance traveled by furthest insertion"
    )

parser.add_argument (
      "-l"
    , "--lkh"
    , action  = "store_true"
    , dest    = "need_lkh"
    , default = False
    , help    = "calculate the distance traveled by lkh"
    )

parser.add_argument (
      "-p"
    , "--print-tours"
    , action  = "store_true"
    , dest    = "need_tours_printed"
    , default = False
    , help    = "print explicit tours"
    )

parser.add_argument (
      "tsp_queue"
    , nargs   = "+"
    , metavar = "PATH"
    , help    = "Path to directory or .tsp file. If PATH is a directory, run "
                "on all .tsp files in the directory."
    )

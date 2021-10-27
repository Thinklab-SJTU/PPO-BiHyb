import numpy as np
import lkh_wrapper


def get_edge_weight(tsp, city1, city2):
    weight = tsp.get_weight(*(city1,city2)) if city1 > city2 else tsp.get_weight(*(city2,city1))
    return weight


def calc_nearest_neighbor_tour_len(tsp):
    path = calc_nearest_neighbor_tour(tsp)
    return_path = path.copy()
    tour_len = 0
    while(len(path) > 1):
        start_node = path.pop()
        next_node  = path[-1]
        tour_len += get_edge_weight(tsp, start_node, next_node)
    return return_path, tour_len


def calc_nearest_neighbor_tour(tsp):
    """Construct a tour through all cities in a TSP by following the nearest neighbor heuristic"""
    nearest_neighbor_path = [0]
    current_city          = 0
    cities_to_travel      = set(range(1, len(list(tsp.get_nodes())) ))

    while cities_to_travel:
        distance_to_current_city = lambda city: get_edge_weight(tsp, city, current_city)
        current_city = min(cities_to_travel, key = distance_to_current_city)
        nearest_neighbor_path.append(current_city)
        cities_to_travel.remove(current_city)
    nearest_neighbor_path.append(nearest_neighbor_path[0])
    
    return nearest_neighbor_path


def calc_lkh_tour_len(tsp, **lkh_kwargs):
    lkh_path = calc_lkh_tour(tsp, **lkh_kwargs)
    return_path = lkh_path.copy()
    for i in range(len(return_path)):
        return_path[i] -= 1
    tour_len = 0
    while(len(lkh_path) > 1):
        start_node = lkh_path.pop()
        next_node  = lkh_path[-1]
        tour_len += get_edge_weight(tsp, start_node-1, next_node-1)
    return return_path, tour_len


def calc_lkh_tour(tsp, **lkh_kwargs):
    solver_path = './LKH-3.0.6/LKH'
    if 'runs' not in lkh_kwargs:
        lkh_kwargs['runs'] = 4
    result = lkh_wrapper.solve(solver_path, problem=tsp, **lkh_kwargs)
    lkh_path = result[0]
    lkh_path.append(lkh_path[0])
    return lkh_path


def calc_furthest_insertion_tour(tsp):
    """Construct a tour through all cities in a TSP by following the furthest insertion heuristic"""
    farest_neighbor_path = [0]
    current_city          = 0
    cities_to_travel      = set(range(1, len(list(tsp.get_nodes())) ))

    while cities_to_travel:
        # selection furthest
        # current_city = min(farest_neighbor_path, key =lambda city_n: max(cities_to_travel, key = lambda city_y: adj[city_n][city_y]))
        current_city = -1
        dist_f = 0
        for c1 in cities_to_travel:
            dist_s = np.inf
            for c2 in farest_neighbor_path:
                dist_s = min(dist_s, get_edge_weight(tsp, c1, c2))
            if(dist_f < dist_s):
                dist_f = dist_s
                current_city = c1
        # print(current_city)
        #insertion minimizes cir + crj - cij, and we insert r between i and j
        if(len(farest_neighbor_path) > 1):
            insert_length_change = lambda ci: get_edge_weight(tsp,farest_neighbor_path[ci],current_city)+ get_edge_weight(tsp, farest_neighbor_path[ci+1], current_city) - get_edge_weight(tsp, farest_neighbor_path[ci], farest_neighbor_path[ci+1])
            insert_i = min(list(range(0,len(farest_neighbor_path)-1)), key = insert_length_change)
            farest_neighbor_path.insert(insert_i,current_city)
        else:
            farest_neighbor_path.append(current_city)
        cities_to_travel.remove(current_city)
    farest_neighbor_path.append(farest_neighbor_path[0])
        
    return farest_neighbor_path


def calc_furthest_insertion_tour_len(tsp):
    path = calc_furthest_insertion_tour(tsp)
    return_path = path.copy()
    tour_len = 0
    while(len(path) > 1):
        start_node = path.pop()
        next_node  = path[-1]
        tour_len += get_edge_weight(tsp, start_node, next_node)
    return return_path, tour_len


def get_adj(problem):
    nodes = list(problem.get_nodes())
    dim = int(problem.dimension)
    
    adj_matrix = np.zeros(dim**2).reshape(dim, dim)
    for i in range(0,dim):
        for j in range(i+1,dim):
            adj_matrix[i][j] = problem.get_weight(*(nodes[i],nodes[j]))
            # distances.euclidean(start, end)
    adj_matrix += adj_matrix.T - np.diag(adj_matrix.diagonal()) #+np.inf
    return adj_matrix


def get_lower_matrix_tsp(problem):
    nodes = list(problem.get_nodes())
    dim = int(problem.dimension)
    
    adj_matrix = []
    for i in range(0,dim):
        matrix_d = []
        for j in range(0,i+1):
            if j == i :
                matrix_d.append(0)
            else:
                matrix_d.append(problem.get_weight(*(nodes[i],nodes[j])))
        adj_matrix.append(matrix_d)
    return adj_matrix


def get_lower_matrix(problem, feas_weight=1, infeas_weight=10):
    dim = int(problem.dimension)

    flat_edge_list = []
    for key in problem.edge_data.keys():
        flat_edge_list.append(key)
        flat_edge_list += problem.edge_data[key]

    edge_list = []
    for i in range(len(flat_edge_list) // 2):
        edge_list.append((flat_edge_list[i * 2]-1, flat_edge_list[i * 2 + 1]-1))

    adj_matrix = []
    for i in range(0, dim):
        matrix_d = []
        for j in range(0, i + 1):
            if (i, j) in edge_list or (j, i) in edge_list:
                matrix_d.append(feas_weight)
            else:
                matrix_d.append(infeas_weight)
        adj_matrix.append(matrix_d)
    return adj_matrix


def get_edge_dict(problem, adj=None):
    nodes = list(problem.get_nodes())
    dim = int(problem.dimension)
    edge_dict = {}
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            if adj is not None and adj[i, j] == 0:
                continue
            edge_dict[(i, j)] = problem.get_weight(nodes[i], nodes[j])
    return edge_dict




def get_matrix_hcp(problem):
    # edge_data = problem.edge_data_format
    edge_list = problem.edge_data[1]
    edge_list = np.append(1,edge_list)
    dim = problem.dimension
    
    adj_matrix = (np.ones(dim**2)*2).reshape(dim, dim)
    for i in range(0,dim):
        adj_matrix.append(np.ones(i+1)*2)
    
    # adj_matrix +=1
    for i in range(0,len(edge_list),2):
        c1 = edge_list[i]-1
        c2 = edge_list[i+1]-1
        adj_matrix[c2][c1] = 1
        adj_matrix[c1][c2] = 1
    return adj_matrix


def solveFarthestInsertion(problem):
    farest_neighbor_path = [0]
    distance = get_adj(problem)
    done          = [0]
    points      = set(range(1, problem.dimension))

    while points:
        farthestPoint = None
        # Find the farthest city to the current cycle
        for current in done:
            for point in points:
                d = distance[current,point]
                if (farthestPoint is None or d > farthestPoint[1]):
                    farthestPoint = (point, d)
        farthestPoint = farthestPoint[0]

        # Find the closest edge of the cycle to the farthest city
        last = None
        best = None
        for current in done:
            if (last is not None):
                d1 = distance[last, farthestPoint]
                d2 = distance[current, farthestPoint]
                d3 = distance[last, current]
                d = d1 + d2 - d3

                if (best is None or d < best[2]):
                    best = (last, current, d)
            last = current

        if (last is None):
            last = done[0]

        d1 = distance[last, farthestPoint]
        d2 = distance[done[0], farthestPoint]
        d3 = distance[last, done[0]]
        d = d1 + d2 - d3
        if (best is None or d < best[2]):
            best = (last, done[0], d)

        # Connect the farthest city to the cycle
        done.insert(done.index(best[0]) + 1, farthestPoint)
        points.remove(farthestPoint)
    
    done.append(0)

    tour_len = 0
    farest_neighbor_path = done.copy()
    while(len(done) > 1):
        start_node = done.pop()
        next_node  = done[-1]
        tour_len += distance[start_node, next_node]
    return farest_neighbor_path, tour_len

import torch
import scipy.optimize as opt
import numpy as np
import random
import os
from multiprocessing import Pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.transforms import OneHotDegree
import torch_geometric as pyg
from utils.sinkhorn import Sinkhorn
from a_star import a_star
import time

VERY_LARGE_INT = 65536


class GEDenv(object):
    def __init__(self, solver_type='hungarian', dataset='AIDS700nef'):
        self.solver_type = solver_type
        self.dataset = dataset
        self.process_dataset()
        self.available_solvers = ('hungarian', 'ipfp', 'rrwm', 'beam')
        assert solver_type in self.available_solvers

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        if self.dataset in ['AIDS700nef', 'LINUX', 'ALKANE', 'IMDBMulti']:  # SimGNN datasets
            from torch_geometric.datasets import GEDDataset
            ori_train = GEDDataset('datasets/{}'.format(self.dataset), self.dataset, train=True)
            self.training_graphs = ori_train[:len(ori_train) // 4 * 3]
            self.val_graphs = ori_train[len(ori_train) // 4 * 3:]
            self.testing_graphs = GEDDataset('datasets/{}'.format(self.dataset), self.dataset, train=False)
        elif self.dataset in ['AIDS-20', 'AIDS-20-30', 'AIDS-30-50', 'AIDS-50-100']:  # gedlib datasets
            from ged_data.gedlib_dataset import GEDDataset
            all_graphs = GEDDataset('datasets/GEDLIB', self.dataset, set='test')
            self.training_graphs = all_graphs[len(all_graphs) // 4:]
            self.val_graphs = all_graphs[:len(all_graphs) // 4]
            self.testing_graphs = all_graphs[:len(all_graphs) // 4]
        else:
            raise ValueError('Unknown dataset name {}'.format(self.dataset))

        self.nged_matrix = self.training_graphs.norm_ged
        self.real_data_size = self.nged_matrix.size(0)

        max_degree = 0
        for g in self.training_graphs + self.val_graphs + self.testing_graphs:
            if g.edge_index.size(1) > 0:
                max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
        one_hot_degree = OneHotDegree(max_degree, cat=self.training_graphs[0].x is not None)
        self.training_graphs.transform = one_hot_degree
        self.val_graphs.transform = one_hot_degree
        self.testing_graphs.transform = one_hot_degree

        self.feature_dim = self.training_graphs.num_features
        self.ori_feature_dim = self.feature_dim - max_degree - 1

    def generate_tuples(self, graphs_dataset, num_samples, rand_id, device):
        random.seed(int(rand_id))
        num_graphs = len(graphs_dataset)
        indices = [random.sample(range(num_graphs), 2) for _ in range(num_samples)]
        ged_results = {}
        for key in self.available_solvers:
            result_path = f'ged_data/cache/{key}_{num_samples}_{self.dataset}_{rand_id}.pt'
            if os.path.exists(result_path):
                result_dict = torch.load(result_path)
            else:
                result_dict = {}
                for idx_1, idx_2 in indices:
                    graph1, graph2 = graphs_dataset[idx_1].to(device), graphs_dataset[idx_2].to(device)
                    sol, _, sec = self.solve_feasible_ged(graph1, graph2, key)
                    result_dict[f'{idx_1},{idx_2}'] = (sol, sec)
                torch.save(result_dict, result_path)
            ged_results[key] = result_dict

        return_tuples = []
        sum_num_nodes = 0
        for i, (idx_1, idx_2) in enumerate(indices):
            graph1, graph2 = graphs_dataset[idx_1].to(device), graphs_dataset[idx_2].to(device)
            ori_k = self.construct_k(graph1, graph2).squeeze(0)
            ged_label = (self.nged_matrix[graph1["i"], graph2["i"]] * 0.5 * (graph1.num_nodes + graph2.num_nodes)).to(device)
            ged_solutions = {}
            ged_times = {}
            for key in self.available_solvers:
                sol, sec = ged_results[key][f'{idx_1},{idx_2}']
                ged_solutions[key] = sol.item()
                ged_times[key] = sec
            if ged_label.item() >= 0:
                ged_solutions.update({'label': ged_label.item()})
            print(f'id {i} {"; ".join([f"{x} ged={ged_solutions[x]:.2f} time={ged_times[x]:.2f}" for x in self.available_solvers])}')
            sum_num_nodes += graph1.num_nodes + graph2.num_nodes
            return_tuples.append((
                graph1,  # input graphs 1
                graph2,  # input graphs 2
                ori_k,  # original quadratic cost matrix k
                ged_solutions[self.solver_type],  # reference GED solution
                ged_solutions,  # all GED solutions
                ged_times,  # GED solving time
            ))

        print(f'average number of nodes: {sum_num_nodes / num_samples / 2}')
        for solver_name in self.available_solvers:
            print(f'{solver_name} ged mean={np.mean([tup[4][solver_name] for tup in return_tuples]):.4f} '
                  f'std={np.std([tup[4][solver_name] for tup in return_tuples]):.4f}')
        return return_tuples

    def step(self, graph_1, graph_2, ori_k, act, prev_solution):
        new_graph_1 = graph_1.clone()
        assert isinstance(act, torch.Tensor)
        act = act.unsqueeze(1)
        edge_index_bool = new_graph_1.edge_index == act
        edge_index_bool_rev = new_graph_1.edge_index == act.flip(dims=(0,))
        found_bool = torch.logical_or(torch.logical_and(edge_index_bool[0], edge_index_bool[1]),
                                      torch.logical_and(edge_index_bool_rev[0], edge_index_bool_rev[1]))
        if torch.any(found_bool):  # delete edge
            new_graph_1.edge_index = new_graph_1.edge_index[:, ~found_bool]
        else:  # add edge
            new_graph_1.edge_index = torch.cat((new_graph_1.edge_index, act, act.flip(dims=(0,))), dim=1)
        new_solution, _, __ = self.solve_feasible_ged(new_graph_1, graph_2, self.solver_type, ori_k=ori_k)
        reward = prev_solution - new_solution
        return reward, new_graph_1, new_solution

    def step_e2e(self, partial_x, ori_k, act, prev_solution):
        new_x = partial_x.clone()
        new_x[act[0], act[1]] = 1
        done = False
        if torch.all(new_x.sum(dim=0)[:-1] == 1):
            new_x[:-1, -1][~new_x.sum(dim=1)[:-1].to(dtype=torch.bool)] = 1
            done = True
        elif torch.all(new_x.sum(dim=1)[:-1] == 1):
            new_x[-1, :-1][~new_x.sum(dim=0)[:-1].to(dtype=torch.bool)] = 1
            done = True
        new_solution = self.comp_ged(new_x, ori_k)
        reward = prev_solution - new_solution
        return reward, new_x, new_solution, done

    def solve_feasible_ged(self, graph_1, graph_2, solver_type, ori_k=None):
        k = self.construct_k(graph_1, graph_2).squeeze(0)
        prev_time = time.time()
        if ori_k is None:
            ori_k = k
        if solver_type == 'hungarian':
            x = hungarian_ged(k, graph_1.num_nodes, graph_2.num_nodes)
        elif solver_type == 'ipfp':
            x = ipfp_ged(k, graph_1.num_nodes, graph_2.num_nodes)
        elif solver_type == 'beam':
            x = astar_ged(k, graph_1.num_nodes, graph_2.num_nodes)
        elif solver_type == 'rrwm':
            x = rrwm_ged(k, graph_1.num_nodes, graph_2.num_nodes)
        elif solver_type == 'ga':
            x = ga_ged(k, graph_1.num_nodes, graph_2.num_nodes)
        else:
            raise NotImplementedError(f'{solver_type} is not implemented.')
        comp_time = time.time() - prev_time
        return self.comp_ged(x, ori_k), ori_k, comp_time

    def construct_k(self, graph_1, graph_2):
        if isinstance(graph_1, pyg.data.Data):
            graph_1 = pyg.data.Batch.from_data_list([graph_1])
        if isinstance(graph_2, pyg.data.Data):
            graph_2 = pyg.data.Batch.from_data_list([graph_2])
        assert graph_1.num_graphs == graph_2.num_graphs
        device = graph_1.x.device

        edge_index_1 = graph_1.edge_index
        edge_index_2 = graph_2.edge_index
        if hasattr(graph_1, 'edge_attr') and hasattr(graph_2, 'edge_attr'):
            edge_attr_1 = graph_1.edge_attr
            edge_attr_2 = graph_2.edge_attr
        else:
            edge_attr_1 = None
            edge_attr_2 = None
        node_1 = graph_1.x
        node_2 = graph_2.x
        batch_1 = graph_1.batch if hasattr(graph_1, 'batch') else torch.tensor((), dtype=torch.long).new_zeros(
            graph_1.num_nodes)
        batch_2 = graph_2.batch if hasattr(graph_2, 'batch') else torch.tensor((), dtype=torch.long).new_zeros(
            graph_2.num_nodes)

        batch_num = graph_1.num_graphs

        ns_1 = torch.bincount(graph_1.batch)
        ns_2 = torch.bincount(graph_2.batch)

        adj_1 = to_dense_adj(edge_index_1, batch=batch_1, edge_attr=edge_attr_1)
        dummy_adj_1 = torch.zeros(adj_1.shape[0], adj_1.shape[1] + 1, adj_1.shape[2] + 1, device=device)
        dummy_adj_1[:, :-1, :-1] = adj_1
        adj_2 = to_dense_adj(edge_index_2, batch=batch_2, edge_attr=edge_attr_2)
        dummy_adj_2 = torch.zeros(adj_2.shape[0], adj_2.shape[1] + 1, adj_2.shape[2] + 1, device=device)
        dummy_adj_2[:, :-1, :-1] = adj_2

        node_1, _ = to_dense_batch(node_1, batch=batch_1)
        node_2, _ = to_dense_batch(node_2, batch=batch_2)

        dummy_node_1 = torch.zeros(adj_1.shape[0], node_1.shape[1] + 1, node_1.shape[-1], device=device)
        dummy_node_1[:, :-1, :] = node_1
        dummy_node_2 = torch.zeros(adj_2.shape[0], node_2.shape[1] + 1, node_2.shape[-1], device=device)
        dummy_node_2[:, :-1, :] = node_2

        k_diag = self.node_metric(dummy_node_1, dummy_node_2)

        mask_1 = torch.zeros_like(dummy_adj_1)
        mask_2 = torch.zeros_like(dummy_adj_2)
        for b in range(batch_num):
            mask_1[b, :ns_1[b] + 1, :ns_1[b] + 1] = 1
            mask_1[b, :ns_1[b], :ns_1[b]] -= torch.eye(ns_1[b], device=mask_1.device)
            mask_2[b, :ns_2[b] + 1, :ns_2[b] + 1] = 1
            mask_2[b, :ns_2[b], :ns_2[b]] -= torch.eye(ns_2[b], device=mask_2.device)

        a1 = dummy_adj_1.reshape(batch_num, -1, 1)
        a2 = dummy_adj_2.reshape(batch_num, 1, -1)
        m1 = mask_1.reshape(batch_num, -1, 1)
        m2 = mask_2.reshape(batch_num, 1, -1)
        k = torch.abs(a1 - a2) * torch.bmm(m1, m2)
        #k[torch.logical_not(torch.bmm(m1, m2).to(dtype=torch.bool))] = VERY_LARGE_INT
        k = k.reshape(batch_num, dummy_adj_1.shape[1], dummy_adj_1.shape[2], dummy_adj_2.shape[1], dummy_adj_2.shape[2])
        k = k.permute([0, 1, 3, 2, 4])
        k = k.reshape(batch_num, dummy_adj_1.shape[1] * dummy_adj_2.shape[1],
                      dummy_adj_1.shape[2] * dummy_adj_2.shape[2])

        for b in range(batch_num):
            k_diag_view = torch.diagonal(k[b])
            k_diag_view[:] = k_diag[b].reshape(-1)

        return k

    def node_metric(self, node1, node2):
        if self.dataset in ['AIDS700nef', 'AIDS-20', 'AIDS-20-30', 'AIDS-30-50', 'AIDS-50-100']:
            node1, node2 = node1[:, :, :self.ori_feature_dim], node2[:, :, :self.ori_feature_dim]
            encoding = torch.sum(torch.abs(node1.unsqueeze(2) - node2.unsqueeze(1)), dim=-1).to(dtype=torch.long)
            mapping = torch.Tensor([0, 1, 1])
        elif self.dataset in ['CMU', 'Willow']:
            assert self.ori_feature_dim == 0
            encoding = torch.sum(torch.abs(node1.unsqueeze(2) - node2.unsqueeze(1)), dim=-1).to(dtype=torch.long)
            mapping = torch.Tensor([0, VERY_LARGE_INT, 0])
        else:
            assert self.ori_feature_dim == 0
            encoding = torch.sum(torch.abs(node1.unsqueeze(2) - node2.unsqueeze(1)), dim=-1).to(dtype=torch.long)
            mapping = torch.Tensor([0, 1, 0])
        return mapping[encoding]

    @staticmethod
    def comp_ged(_x, _k):
        if len(_x.shape) == 3 and len(_k.shape) == 3:
            _batch = _x.shape[0]
            return torch.bmm(torch.bmm(_x.reshape(_batch, 1, -1), _k), _x.reshape(_batch, -1, 1)).view(_batch)
        elif len(_x.shape) == 2 and len(_k.shape) == 2:
            return torch.mm(torch.mm(_x.reshape( 1, -1), _k), _x.reshape( -1, 1)).view(1)
        else:
            raise ValueError('Input dimensions not supported.')


def ipfp_ged(k, n1, n2, max_iter=100):
    assert len(k.shape) == 2
    v = hungarian_ged(k, n1, n2)
    #v = torch.ones(n1 + 1, n2 + 1, dtype=k.dtype, device=k.device) / (n1 + 1) / (n2 + 1)
    last_v = v
    best_binary_sol = v
    n1, n2 = torch.tensor([n1], device=k.device), torch.tensor([n2], device=k.device)

    #k_diag = torch.diag(k)
    #k_offdiag = k - torch.diag(k_diag)
    best_upper_bound = float('inf')

    for i in range(max_iter):
        cost = torch.mm(k, v.view(-1, 1)).reshape(n1+1, n2+1)
        binary_sol = hungarian_lap(cost, n1, n2)[0]
        upper_bound = GEDenv.comp_ged(binary_sol, k)
        if upper_bound < best_upper_bound:
            best_binary_sol = binary_sol
            best_upper_bound = upper_bound
        alpha = torch.mm(torch.mm(v.view(1, -1), k), (binary_sol - v).view(-1, 1)) #+ \
                #torch.mm(k_diag.view(1, -1), (binary_sol - v).view(-1, 1))
        beta = GEDenv.comp_ged(binary_sol - v, k)
        t0 = - alpha / beta
        if beta <= 0 or t0 >= 1:
            v = binary_sol
        else:
            v = v + t0 * (binary_sol - v)
        last_v_sol = GEDenv.comp_ged(last_v, k)
        if torch.abs(last_v_sol - torch.mm(cost.view(1, -1), binary_sol.view(-1, 1))) / last_v_sol < 1e-3:
            break
        last_v = v

    pred_x = best_binary_sol
    return pred_x


def astar_ged(k, n1, n2, beamwidth=1):
    x_pred, tree_size = a_star(
        k.unsqueeze(0), np.array([n1]), np.array([n2]),
        heuristic_prediction_hun,
        beam_width=beamwidth,
        trust_fact=1.,
        no_pred_size=0,
    )
    return x_pred.squeeze(0)


def hungarian_ged(k, n1, n2):
    x, _ = heuristic_prediction_hun(k, n1, n2)
    return x


def heuristic_prediction_hun(k, n1, n2, partial_pmat=None):
    assert len(k.shape) == 2
    k_prime = k.reshape(-1, n1+1, n2+1)
    node_costs = torch.empty(k_prime.shape[0], device=k.device)
    for i in range(k_prime.shape[0]):
        _, node_costs[i] = hungarian_lap(k_prime[i], n1, n2)

    node_cost_mat = node_costs.reshape(n1+1, n2+1)
    if partial_pmat is None:
        partial_pmat = torch.zeros_like(node_cost_mat)
    graph_1_mask = ~partial_pmat.sum(dim=-1).to(dtype=torch.bool)
    graph_2_mask = ~partial_pmat.sum(dim=-2).to(dtype=torch.bool)
    graph_1_mask[-1] = 1
    graph_2_mask[-1] = 1
    node_cost_mat = node_cost_mat[graph_1_mask, :]
    node_cost_mat = node_cost_mat[:, graph_2_mask]

    x, lb = hungarian_lap(node_cost_mat, torch.sum(graph_1_mask[:-1]), torch.sum(graph_2_mask[:-1]))
    return x, lb


def hungarian_lap(node_cost_mat, n1, n2):
    assert node_cost_mat.shape[-2] == n1+1
    assert node_cost_mat.shape[-1] == n2+1
    device = node_cost_mat.device
    upper_left = node_cost_mat[:n1, :n2]
    upper_right = torch.full((n1, n1), float('inf'), device=device)
    torch.diagonal(upper_right)[:] = node_cost_mat[:-1, -1]
    lower_left = torch.full((n2, n2), float('inf'), device=device)
    torch.diagonal(lower_left)[:] = node_cost_mat[-1, :-1]
    lower_right = torch.zeros((n2, n1), device=device)

    large_cost_mat = torch.cat((torch.cat((upper_left, upper_right), dim=1),
                                torch.cat((lower_left, lower_right), dim=1)), dim=0)

    large_pred_x = hungarian(-large_cost_mat)
    pred_x = torch.zeros_like(node_cost_mat)
    pred_x[:n1, :n2] = large_pred_x[:n1, :n2]
    pred_x[:-1, -1] = torch.sum(large_pred_x[:n1, n2:], dim=1)
    pred_x[-1, :-1] = torch.sum(large_pred_x[n1:, :n2], dim=0)

    ged_lower_bound = torch.sum(pred_x * node_cost_mat)

    return pred_x, ged_lower_bound


def hungarian(s: torch.Tensor, n1=None, n2=None, mask=None, nproc=1):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :param nproc: number of parallel processes (default =1 for no parallel)
    :return: optimal permutation matrix
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood.')

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num
    if mask is None:
        mask = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([hung_kernel(perm_mat[b], n1[b], n2[b], mask[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat


def hung_kernel(s: torch.Tensor, n1=None, n2=None, mask=None):
    if mask is None:
        if n1 is None:
            n1 = s.shape[0]
        if n2 is None:
            n2 = s.shape[1]
        row, col = opt.linear_sum_assignment(s[:n1, :n2])
    else:
        mask = mask.cpu()
        s_mask = s[mask]
        if s_mask.size > 0:
            dim0 = torch.sum(mask, dim=0).max()
            dim1 = torch.sum(mask, dim=1).max()
            row, col = opt.linear_sum_assignment(s_mask.reshape(dim0, dim1))
            row = torch.nonzero(torch.sum(mask, dim=1), as_tuple=True)[0][row]
            col = torch.nonzero(torch.sum(mask, dim=0), as_tuple=True)[0][col]
        else:
            row, col = [], []
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat


def ga_ged(k, n1, n2, max_iter=100, sk_iter=10, tau0=1, tau_min=0.1, gamma=0.95):
    assert len(k.shape) == 2
    v = torch.ones(n1+1, n2+1, dtype=k.dtype, device=k.device) / (n1 + 1) / (n2 + 1)
    #v = hungarian_ged(k, n1, n2)
    v = v.reshape(-1, 1)
    n1, n2 = torch.tensor([n1], device=k.device), torch.tensor([n2], device=k.device)
    tau = tau0

    while tau >= tau_min:
        for i in range(max_iter):
            last_v = v
            v = torch.mm(k, v)
            v = Sinkhorn(max_iter=sk_iter, tau=tau)(-v.view(n1+1, -1), n1+1, n2+1).reshape(-1, 1)
            if torch.norm(v - last_v) < 1e-4:
                break
        tau = tau * gamma

    for i in range(max_iter):
        last_v = v
        v = torch.mm(k, v)
        v, _ = hungarian_lap(-v.reshape(n1 + 1, -1), n1, n2)
        v = v.view(-1, 1)
        if torch.norm(v - last_v) < 1e-4:
            break
    pred_x = v.reshape(n1+1, -1)
    return pred_x


def rrwm_ged(k, n1, n2, max_iter=100, sk_iter=100, alpha=0.2, beta=100):
    assert len(k.shape) == 2
    d = k.sum(dim=1, keepdim=True)
    dmax = d.max(dim=0, keepdim=True).values
    k = k / (dmax + d.min() * 1e-5)

    #v = torch.ones(n1+1, n2+1, dtype=k.dtype, device=k.device) / (n1 + 1) / (n2 + 1)
    v = hungarian_ged(k, n1, n2)
    v = v.reshape(-1, 1)
    n1, n2 = torch.tensor([n1], device=k.device), torch.tensor([n2], device=k.device)

    for i in range(max_iter):
        last_v = v
        v = torch.mm(k, v)
        n = torch.norm(v, p=1, dim=0, keepdim=True)
        v = v / n
        v = alpha * Sinkhorn(max_iter=sk_iter, tau=v.max() / beta)(-v.view(n1+1, -1), n1+1, n2+1).reshape(-1, 1) + \
            (1 - alpha) * last_v
        n = torch.norm(v, p=1, dim=0, keepdim=True)
        v = torch.matmul(v, 1 / n)

        if torch.norm(v - last_v) < 1e-3:
            break

    pred_x, lb = hungarian_lap(-v.view(n1 + 1, -1), n1, n2)
    return pred_x

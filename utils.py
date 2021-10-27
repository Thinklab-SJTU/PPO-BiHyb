import networkx as nx
import torch_geometric as pyg
import torch
from texttable import Texttable
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import itertools


def construct_graph_batch(graphs, one_hot_degree, device):
    # build pyg data
    if isinstance(graphs, nx.DiGraph):
        pyg_data_list = [nx_to_pyg(graphs, one_hot_degree, device)]
    elif isinstance(graphs, pyg.data.Data):
        pyg_data_list = [pyg_transform(graphs, one_hot_degree, device)]
    elif isinstance(graphs, list) and isinstance(graphs[0], nx.DiGraph):
        pyg_data_list = [nx_to_pyg(x, one_hot_degree, device) for x in graphs]
    elif isinstance(graphs, list) and isinstance(graphs[0], pyg.data.Data):
        pyg_data_list = [pyg_transform(x, one_hot_degree, device) for x in graphs]
    else:
        raise ValueError('Data type not understood')

    # put graph_batch.batch to device
    graph_batch = pyg.data.Batch.from_data_list(pyg_data_list)
    graph_batch.batch = graph_batch.batch.to(graph_batch.x.device)
    return graph_batch


def reverse_pyg_graph(graph):
    rgraph = graph.clone()
    rgraph.edge_index[0, :] = rgraph.edge_index[1, :]
    rgraph.edge_index[1, :] = rgraph.edge_index[0, :]
    return rgraph


def nx_to_pyg(nx_graph: nx.Graph, onehot_degree: int = 0, device: torch.device = None):
    mapping = {name: j for j, name in enumerate(nx_graph.nodes())}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
    node_feat = torch.tensor([v['features'] for k,v in nx_graph.nodes(data=True)], dtype=torch.float32)

    pyg_graph = pyg.data.Data(x=node_feat, edge_index=edge_index)
    return pyg_transform(pyg_graph, onehot_degree, device)


def pyg_transform(pyg_graph: pyg.data.Data, onehot_degree: int = 0, device: torch.device = None):
    if device is not None:
        pyg_graph = pyg_graph.to(device)

    if onehot_degree > 0:
        pyg_graph = pyg.transforms.OneHotDegree(onehot_degree, in_degree=True, cat=True)(pyg_graph)

    return pyg_graph


def print_args(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def pad_tensor(inp):
    """
    Pad a list of input tensors into a list of tensors with same dimension
    :param inp: input tensor list
    :return: output tensor list
    """
    assert type(inp[0]) == torch.Tensor
    it = iter(inp)
    t = next(it)
    max_shape = list(t.shape)
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in inp:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(torch.nn.functional.pad(t, pad_pattern, 'constant', 0))

    return padded_ts


def random_triangulate(n):
    """
    Output a randomly generated triangulation by performing delaunay triangulation on a random point set.
    :param n: number of nodes
    :return: adjacency matrix A
    """
    P = np.random.random((n, 2))
    if n < 3:
        A = fully_connect(P)
    else:
        try:
            d = Delaunay(P)
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print('Delaunay triangulation error detected. Return fully-connected graph.')
            print('Traceback:')
            print(err)
            A = fully_connect(P)
    return A


def fully_connect(P: np.ndarray):
    """
    fully connect a graph
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    return A
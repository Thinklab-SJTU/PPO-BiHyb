import os
import os.path as osp
import glob
import pickle
import re
import itertools

import torch
import torch.nn.functional as F
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip, extract_tar)
from torch_geometric.utils import to_undirected


class GEDDataset(InMemoryDataset):
    r"""The GED datasets from the `"Graph Edit Distance Computation via Graph
    Neural Networks" <https://arxiv.org/abs/1808.05689>`_ paper.
    GEDs can be accessed via the global attributes :obj:`ged` and
    :obj:`norm_ged` for all train/train graph pairs and all train/test graph
    pairs:
    .. code-block:: python
        dataset = GEDDataset(root, name="LINUX")
        data1, data2 = dataset[0], dataset[1]
        ged = dataset.ged[data1.i, data2.i]  # GED between `data1` and `data2`.
    .. note::
        :obj:`ALKANE` is missing GEDs for train/test graph pairs since they are
        not provided in the `official datasets
        <https://github.com/yunshengb/SimGNN>`_.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"AIDS700nef"`,
            :obj:`"LINUX"`, :obj:`"ALKANE"`, :obj:`"IMDBMulti"`).
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://github.com/dbblumenthal/gedlib/trunk/data/datasets/{}'
    datasets = {
        'AIDS-10-16': {
            'data_dir': 'AIDS/data',
            'min_nodes': 10,
            'max_nodes': 16,
            'count': 4,
        },
        'AIDS-20': {
            'data_dir': 'AIDS/data',
            'min_nodes': 20,
            'max_nodes': 20,
            'count': 25,
        },
        'AIDS-20-30': {
            'data_dir': 'AIDS/data',
            'min_nodes': 20,
            'max_nodes': 30,
            'count': 25,
        },
        'AIDS-30-50': {
            'data_dir': 'AIDS/data',
            'min_nodes': 30,
            'max_nodes': 50,
            'count': 25,
        },
        'AIDS-50-100': {
            'data_dir': 'AIDS/data',
            'min_nodes': 50,
            'max_nodes': 100,
            'count': 25,
        },
    }

    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te',
    ]

    def __init__(self, root, name, set='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        assert self.name in self.datasets.keys()
        super(GEDDataset, self).__init__(root, transform, pre_transform,
                                         pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)
        self.norm_ged = torch.full((len(self), len(self)), -1, dtype=torch.float)

    @property
    def raw_file_names(self):
        return [self.datasets[self.name]['data_dir']]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.name)]

    def download(self):
        if all([osp.exists(osp.join(self.raw_dir, p)) for p in self.raw_file_names]):
            pass
        else:
            retcode = os.system('svn checkout ' + self.url.format(self.datasets[self.name]['data_dir']) +
                                ' {}/{}'.format(self.raw_dir, self.datasets[self.name]['data_dir']))
            assert retcode == 0

    def process(self):
        Ns = []
        assert len(self.raw_paths) == 1
        assert len(self.processed_paths) == 1
        paths = glob.glob(osp.join(self.raw_paths[0], '*.gxl'))
        names = sorted([i.split(os.sep)[-1] for i in paths], key=lambda x: re.findall(r'[0-9]+', x)[0])

        data_list = []
        i_offset = 0
        num_nodes_count = np.zeros(self.datasets[self.name]['max_nodes'] - self.datasets[self.name]['min_nodes'] + 1, dtype=np.int)
        for i, name in enumerate(names):
            G = read_gxl(osp.join(self.raw_paths[0], name))
            mapping = {name: j for j, name in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

            if G.number_of_nodes() < self.datasets[self.name]['min_nodes'] or \
                G.number_of_nodes() > self.datasets[self.name]['max_nodes']:
                print('skipping graph {} due to number of nodes {}'.format(i, G.number_of_nodes()))
                i_offset += 1
                continue

            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            if edge_index.numel() == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_index = to_undirected(edge_index, num_nodes=G.number_of_nodes())

            data = Data(edge_index=edge_index, i=i-i_offset)
            data.num_nodes = G.number_of_nodes()

            if 'AIDS' in self.name:
                x = torch.zeros(data.num_nodes, dtype=torch.long)
                try:
                    for node, sym in G.nodes(data='symbol'):
                        sym = sym.strip()
                        assert sym in self.types
                        x[int(node)] = self.types.index(sym)
                except AssertionError:
                    print('skipping graph {} due to unknwon molecule {}'.format(i, sym))
                    i_offset += 1
                    continue
                data.x = F.one_hot(x, num_classes=len(self.types)).to(torch.float)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if num_nodes_count[G.number_of_nodes() - self.datasets[self.name]['min_nodes']] >= self.datasets[self.name]['count']:
                print('skipping graph {} because there are too many graphs with {} nodes'.format(i, G.number_of_nodes()))
                i_offset += 1
                continue
            else:
                num_nodes_count[G.number_of_nodes() - self.datasets[self.name]['min_nodes']] += 1

            Ns.append(G.number_of_nodes())
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


dtype_mapping = {
    'double': float,
    'float': float,
    'string': str,
    'int': int,
}


def read_gxl(gxl_path):
    f = open(gxl_path, 'rb')
    tree_gxl = ET.parse(f)
    root_gxl = tree_gxl.getroot()
    edgedefault = root_gxl.find('graph').get('edgemode', None)
    if edgedefault == 'directed':
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    node_id = []
    # Parse nodes
    for i, node in enumerate(root_gxl.iter('node')):
        node_id += [node.get('id')]
        attr_dict = dict()
        for node_attr in node.iter('attr'):
            dtype = dtype_mapping[node_attr.find('*').tag.lower()]
            attr_dict[node_attr.get('name')] = dtype(node_attr.find('*').text)
        G.add_node(node.get('id'), **attr_dict)

    node_id = np.array(node_id)

    ##Parsing edges
    for edge in root_gxl.iter('edge'):
        s = np.where(node_id==edge.get('from'))[0][0]
        t = np.where(node_id==edge.get('to'))[0][0]

        attr_dict = dict()
        for edge_attr in edge.iter('attr'):
            dtype = dtype_mapping[edge_attr.find('*').tag]
            attr_dict[edge_attr.get('name')] = dtype(edge_attr.find('*').text)

        #Add edge with original node names and nlr value to graph
        G.add_edge(node_id[s], node_id[t], **attr_dict)

    return G
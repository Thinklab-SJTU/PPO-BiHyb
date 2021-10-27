import torch
from torch import nn
import torch_geometric as pyg
from torch_scatter import scatter
from itertools import chain


class ResNetBlock(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_layers=3, batch_norm=True):
        super(ResNetBlock, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_layers = num_layers

        self.first_linear = None
        self.last_linear = None
        self.sequential = []
        self.output_seq = []

        for l in range(self.num_layers):
            if l == 0:
                self.first_linear = nn.Linear(self.num_in_feats, self.num_out_feats)
                if batch_norm: self.sequential.append(nn.BatchNorm1d(self.num_out_feats))
                self.sequential.append(nn.ReLU())
            elif l == self.num_layers - 1:
                self.last_linear = nn.Linear(self.num_out_feats, self.num_out_feats)
                if batch_norm: self.output_seq.append(nn.BatchNorm1d(self.num_out_feats))
            else:
                self.sequential.append(nn.Linear(self.num_out_feats, self.num_out_feats))
                if batch_norm: self.sequential.append(nn.BatchNorm1d(self.num_out_feats))
                self.sequential.append(nn.ReLU())

        self.sequential = nn.Sequential(*self.sequential)
        self.output_seq = nn.Sequential(*self.output_seq)

        self.init_parameters()

    def init_parameters(self):
        for mod in chain(self.sequential, self.output_seq):
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)

    def forward(self, inp):
        x1 = self.first_linear(inp)
        x2 = self.sequential(x1) + x1
        return self.output_seq(x2)


class GCN(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_layers=3, batch_norm=True):
        super(GCN, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_layers = num_layers

        for l in range(self.num_layers):
            if l == 0:
                conv = pyg.nn.GCNConv(self.num_in_feats, self.num_out_feats)
            else:
                conv = pyg.nn.GCNConv(self.num_out_feats, self.num_out_feats)
            if batch_norm:
                norm = nn.BatchNorm1d(self.num_out_feats)
            else:
                norm = nn.Identity()
            self.add_module('conv_{}'.format(l), conv)
            self.add_module('norm_{}'.format(l), norm)

        self.init_parameters()

    def init_parameters(self):
        for l in range(self.num_layers):
            nn.init.xavier_uniform_(getattr(self, 'conv_{}'.format(l)).weight)

    def forward(self, *args):
        if len(args) == 1 and isinstance(args[0], (pyg.data.Data, pyg.data.Batch)):
            x, edge_index = args[0].x, args[0].edge_index
        elif len(args) == 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            x, edge_index = args
        else:
            raise ValueError('Unknown combination of data types: {}'.format(','.join([type(x) for x in args])))
        for l in range(self.num_layers):
            conv = getattr(self, 'conv_{}'.format(l))
            norm = getattr(self, 'norm_{}'.format(l))
            x = conv(x, edge_index)
            x = nn.functional.relu(norm(x))
        return x


class GraphAttentionPooling(nn.Module):
    """
    Attention module to extract global feature of a graph.
    """
    def __init__(self, feat_dim):
        """
        :param feat_dim: number dimensions of input features.
        """
        super(GraphAttentionPooling, self).__init__()
        self.feat_dim = feat_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.feat_dim, self.feat_dim))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param batch: Batch vector, which assigns each node to a specific example
        :return representation: A graph level representation matrix.
        """
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter(x, batch, dim=0, dim_size=size, reduce='mean')
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        coefs = torch.sigmoid((x * transformed_global[batch] * 10).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return scatter(weighted, batch, dim=0, dim_size=size, reduce='add')

    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))


class AttentionModule(nn.Module):
    """
    Attention module for feature query
    """
    def __init__(self, feat_dim_query, feat_dim_val):
        """
        :param feat_dim_query: number dimensions of query features.
        :param feat_dim_val: number of dimensions of value features.
        """
        super(AttentionModule, self).__init__()
        self.feat_dim_query = feat_dim_query
        self.feat_dim_val = feat_dim_val
        self.hidden_dim = max(feat_dim_query, feat_dim_val)
        self.weight_query = nn.Parameter(torch.Tensor(self.feat_dim_query, self.hidden_dim))
        self.weight_val = nn.Parameter(torch.Tensor(self.feat_dim_val, self.hidden_dim))

        self.init_parameters()

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_query)
        torch.nn.init.xavier_uniform_(self.weight_val)

    def forward(self, query, value):
        """
        :param query: query tensor (batch, feat_dim_query)
        :param value: value tensor (batch, num_nodes, feat_dim_query)
        """
        assert len(query.shape) == 2
        assert len(value.shape) == 3

        trans_q = torch.mm(query, self.weight_query).unsqueeze(1)
        trans_v = torch.matmul(value, self.weight_val)

        sim_weights = torch.tanh(torch.sum(trans_q * trans_v, dim=-1))

        return sim_weights


class TensorNetworkModule(torch.nn.Module):
    """
    Tensor Network module to calculate similarity vector.
    """
    def __init__(self, input_features, tensor_neurons):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.input_features = input_features
        self.tensor_neurons = tensor_neurons
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_features, self.input_features, self.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 2*self.input_features))
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = len(embedding_1)
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.input_features, -1))
        scoring = scoring.view(batch_size, self.input_features, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.input_features, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = torch.relu(scoring + block_scoring + self.bias.view(-1))
        return scores

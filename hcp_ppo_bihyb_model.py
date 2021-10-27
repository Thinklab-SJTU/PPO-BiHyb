import torch
from torch import nn
from pyg_graph_models import GCN, GraphAttentionPooling, ResNetBlock, TensorNetworkModule
from utils import construct_graph_batch
from torch_geometric.utils import to_dense_batch
from torch.distributions import Categorical
import torch_geometric as pyg


def matrix_list_to_graphs(lower_left_matrices, device):
    graphs = []
    #edge_candidates = []
    for b, lower_left_m in enumerate(lower_left_matrices):
        edge_indices = [[], []]
        edge_attrs = [] ###############################
        x = torch.ones(len(lower_left_m), 1)
        #edge_cand = {x: set() for x in range(len(lower_left_m))}
        for row, cols in enumerate(lower_left_m):
            for col, weight in enumerate(cols):
                if weight == 0 or weight >= 2:
                    pass
                else:
                    edge_indices[0].append(row)
                    edge_indices[1].append(col)
                    edge_attrs.append(weight)  #######################
                    x[row] += weight
                    x[col] += weight
                    #edge_cand[row].add(col)
                    #edge_cand[col].add(row)
        edge_indices = torch.tensor(edge_indices)
        edge_attrs = torch.Tensor(edge_attrs).to(device)  ####################
        #x = (x) / torch.std(x)
        # graphs.append(pyg.data.Data(x=x, edge_index=edge_indices)) #, edge_attrs=edge_attrs)) ############################
        graphs.append(pyg.data.Data(x=x, edge_index=edge_indices, edge_attrs=edge_attrs))  ##########################
        #edge_candidates.append(edge_cand)
    return graphs #, edge_candidates


class GraphEncoder(torch.nn.Module):
    def __init__(
            self,
            node_feature_dim,
            node_output_size,
            batch_norm,
            one_hot_degree,
            num_layers=10
    ):
        super(GraphEncoder, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_output_size = node_output_size
        self.one_hot_degree = one_hot_degree
        self.batch_norm = batch_norm
        self.num_layers = num_layers

        one_hot_dim = self.one_hot_degree + 1 if self.one_hot_degree > 0 else 0
        self.siamese_gcn = GCN(self.node_feature_dim + one_hot_dim, self.node_output_size, num_layers=self.num_layers,
                               batch_norm=self.batch_norm)
        self.att = GraphAttentionPooling(self.node_output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, inp_lower_matrix):
        # construct graph batches
        batched_graphs = construct_graph_batch(matrix_list_to_graphs(inp_lower_matrix, self.device),
                                               self.one_hot_degree, self.device)

        # forward pass
        batched_node_feat = self.siamese_gcn(batched_graphs)
        node_feat_reshape, _ = to_dense_batch(batched_node_feat, batched_graphs.batch)
        graph_feat = self.att(batched_node_feat, batched_graphs.batch)
        state_feat = torch.cat(
            (node_feat_reshape, graph_feat.unsqueeze(1).expand(-1, node_feat_reshape.shape[1], -1)), dim=-1)

        return state_feat


class ActorNet(torch.nn.Module):
    def __init__(
            self,
            state_feature_size,
            batch_norm,
    ):
        super(ActorNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.act1_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)
        #self.act2_resnet = ResNetBlock(self.state_feature_size * 2, 1, batch_norm=self.batch_norm)
        self.act2_query = nn.Linear(self.state_feature_size, self.state_feature_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_feat, edge_candidates, known_action=None):
        return self._act(input_feat, edge_candidates, known_action)

    def _act(self, input_feat, edge_candidates, known_action=None):
        if known_action is None:
            known_action = (None, None)
        # roll-out 2 acts
        mask1, ready_nodes1 = self._get_mask1(input_feat.shape[0], input_feat.shape[1], edge_candidates)
        act1, log_prob1, entropy1 = self._select_node(input_feat, mask1, known_action[0])
        mask2, ready_nodes2 = self._get_mask2(input_feat.shape[0], input_feat.shape[1], edge_candidates, act1)
        act2, log_prob2, entropy2 = self._select_node(input_feat, mask2, known_action[1], act1)
        return torch.stack((act1, act2)), torch.stack((log_prob1, log_prob2)), entropy1 + entropy2

    def _select_node(self, state_feat, mask, known_cur_act=None, prev_act=None, greedy_sel_num=0):
        # neural net prediction
        if prev_act is None:  # for act 1
            act_scores = self.act1_resnet(state_feat).squeeze(-1)
        else:  # for act 2
            prev_node_feat = state_feat[torch.arange(len(prev_act)), prev_act, :]
            #state_feat = torch.cat(
            #    (state_feat, prev_node_feat.unsqueeze(1).expand(-1, state_feat.shape[1], -1)), dim=-1)
            #act_scores = self.act2_resnet(state_feat).squeeze(-1)
            act_query = torch.tanh(self.act2_query(prev_node_feat))
            act_scores = (act_query.unsqueeze(1) * state_feat).sum(dim=-1)

        # select action
        act_probs = nn.functional.softmax(act_scores + mask, dim=1)
        if greedy_sel_num > 0:
            argsort_prob = torch.argsort(act_probs, dim=-1, descending=True)
            acts = argsort_prob[:, :greedy_sel_num]
            return acts, act_probs[torch.arange(acts.shape[0]).unsqueeze(-1), acts]
        else:
            dist = Categorical(probs=act_probs)
            if known_cur_act is None:
                act = dist.sample()
                return act, dist.log_prob(act), dist.entropy()
            else:
                return known_cur_act, dist.log_prob(known_cur_act), dist.entropy()

    def _get_mask1(self, batch_size, num_nodes, edge_candidates):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            for node, candidates in edge_candidates[b].items():
                if len(candidates) == 0:
                    pass
                else:
                    masks[b, node] = 0
                    ready_nodes[b].append(node)
        return masks, ready_nodes

    def _get_mask2(self, batch_size, num_nodes, edge_candidates, act1):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            candidates = edge_candidates[b][act1[b].item()]
            for index in candidates:
                masks[b, index] = 0.0
                ready_nodes[b].append(index)
        return masks, ready_nodes


class CriticNet(torch.nn.Module):
    def __init__(
            self,
            state_feature_size,
            batch_norm,
    ):
        super(CriticNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.critic_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, state_feat):
        return self._eval(state_feat)

    def _eval(self, state_feat):
        # get global features
        state_feat = torch.max(state_feat, dim=1).values
        state_value = self.critic_resnet(state_feat).squeeze(-1)
        return state_value

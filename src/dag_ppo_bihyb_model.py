import torch
from torch import nn
from src.pyg_graph_models import GCN, GraphAttentionPooling, ResNetBlock
from utils.utils import construct_graph_batch, reverse_pyg_graph
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch.distributions import Categorical


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
        self.batch_norm = batch_norm
        self.one_hot_degree = one_hot_degree
        self.num_layers = num_layers

        one_hot_dim = self.one_hot_degree + 1 if self.one_hot_degree > 0 else 0
        self.forward_gnn = GCN(self.node_feature_dim + one_hot_dim, self.node_output_size, num_layers=self.num_layers,
                               batch_norm=self.batch_norm)
        self.reverse_gnn = GCN(self.node_feature_dim + one_hot_dim, self.node_output_size, num_layers=self.num_layers,
                               batch_norm=self.batch_norm)
        self.att = GraphAttentionPooling(self.node_output_size * 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_graphs):
        # construct graph batches
        batched_graphs = construct_graph_batch(input_graphs, self.one_hot_degree, self.device)

        # forward pass
        forward_node_feat = self.forward_gnn(batched_graphs)

        # reverse pass
        reversed_graphs = reverse_pyg_graph(batched_graphs)
        reverse_node_feat = self.reverse_gnn(reversed_graphs)

        # attention for global features
        node_feat = torch.cat((forward_node_feat, reverse_node_feat), dim=1)
        node_feat = node_feat / torch.norm(node_feat, dim=-1, keepdim=True)
        graph_feat = self.att(node_feat, batched_graphs.batch)

        # transform to dense batch
        node_feat_reshape, _ = to_dense_batch(node_feat, batched_graphs.batch)
        state_feat = torch.cat(
            (node_feat_reshape, graph_feat.unsqueeze(1).expand(-1, node_feat_reshape.shape[1], -1)), dim=-1)

        return state_feat


class ActorNet(torch.nn.Module):
    def __init__(
            self,
            dag_model,
            state_feature_size,
            batch_norm,
    ):
        super(ActorNet, self).__init__()
        self.dag_model = dag_model
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.act1_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)
        self.act2_resnet = ResNetBlock(self.state_feature_size * 2, 1, batch_norm=self.batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_graph, edge_candidates, known_action=None, greedy_select=False, epsilon=0.0):
        return self._act(input_graph, edge_candidates, known_action, greedy_select, epsilon)

    def _act(self, state_feat, edge_candidates, known_action=None, greedy_select=False, epsilon=0.0):
        if known_action is None:
            known_action = (None, None)
        # roll-out 2 acts
        mask1, ready_nodes1 = self._get_mask1(state_feat.shape[0], state_feat.shape[1], edge_candidates)
        act1, log_prob1, entropy1 = self._select_node(state_feat, mask1, known_action[0])
        mask2, ready_nodes2 = self._get_mask2(state_feat.shape[0], state_feat.shape[1], edge_candidates, act1)
        act2, log_prob2, entropy2 = self._select_node(state_feat, mask2, known_action[1], act1)
        return torch.stack((act1, act2)), torch.stack((log_prob1, log_prob2)), entropy1 + entropy2

    def _select_node(self, state_feat, mask, known_cur_act=None, prev_act=None, greedy_sel_num=0):
        # neural net prediction
        if prev_act is None:  # for act 1
            act_scores = self.act1_resnet(state_feat).squeeze(-1)
        else:  # for act 2
            prev_node_feat = state_feat[torch.arange(len(prev_act)), prev_act, :]
            state_feat = torch.cat(
                (state_feat, prev_node_feat.unsqueeze(1).expand(-1, state_feat.shape[1], -1)), dim=-1)
            act_scores = self.act2_resnet(state_feat).squeeze(-1)

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

    @staticmethod
    def _get_act1_candidates(edge_candidates):
        acts = []
        for node, candidates in edge_candidates.items():
            if len(candidates) > 0:
                acts.append(node)
        return acts

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
            dag_model,
            state_feature_size,
            batch_norm,
    ):
        super(CriticNet, self).__init__()
        self.dag_model = dag_model
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

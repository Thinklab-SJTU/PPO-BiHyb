import torch
from torch import nn
from src.pyg_graph_models import GCN, GraphAttentionPooling, ResNetBlock, TensorNetworkModule
from utils.utils import construct_graph_batch
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch.distributions import Categorical
from utils.sinkhorn import Sinkhorn


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
        self.siamese_gcn = GCN(self.node_feature_dim + one_hot_dim, self.node_output_size, num_layers=self.num_layers,
                               batch_norm=self.batch_norm)
        self.sinkhorn = Sinkhorn(max_iter=20, tau=0.005)
        self.att = GraphAttentionPooling(self.node_output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_graphs_1, input_graphs_2):
        # construct graph batches
        batched_graphs_1 = construct_graph_batch(input_graphs_1, self.one_hot_degree, self.device)
        batched_graphs_2 = construct_graph_batch(input_graphs_2, self.one_hot_degree, self.device)

        # forward pass
        batched_node_feat_1 = self.siamese_gcn(batched_graphs_1)
        batched_node_feat_2 = self.siamese_gcn(batched_graphs_2)

        # compute cross-graph similarity
        node_feat_1, node_indicator_1 = to_dense_batch(batched_node_feat_1, batched_graphs_1.batch)
        node_feat_2, node_indicator_2 = to_dense_batch(batched_node_feat_2, batched_graphs_2.batch)
        num_nodes_1 = node_indicator_1.sum(-1)
        num_nodes_2 = node_indicator_2.sum(-1)
        sim_mat = torch.bmm(node_feat_1, node_feat_2.transpose(1, 2))
        sim_mat = self.sinkhorn(sim_mat, num_nodes_1, num_nodes_2)

        # compute cross-graph difference features
        diff_feat = node_feat_1 - torch.bmm(sim_mat, node_feat_2)

        global_feat_1 = self.att(batched_node_feat_1, batched_graphs_1.batch)
        global_feat_2 = self.att(batched_node_feat_2, batched_graphs_2.batch)

        return diff_feat, global_feat_1, global_feat_2


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

    def forward(self, input_feat, known_action=None):
        return self._act(input_feat, known_action)

    def _act(self, input_feat, known_action=None):
        if known_action is None:
            known_action = (None, None)
        # roll-out 2 acts
        act1, log_prob1, entropy1 = self._select_node(input_feat, known_action[0])
        act2, log_prob2, entropy2 = self._select_node(input_feat, known_action[1], act1)
        return torch.stack((act1, act2)), torch.stack((log_prob1, log_prob2)), entropy1 + entropy2

    def _select_node(self, state_feat, known_cur_act=None, prev_act=None, greedy_sel_num=0):
        # neural net prediction
        if prev_act is None:  # for act 1
            act_scores = self.act1_resnet(state_feat).squeeze(-1)
            mask = torch.zeros_like(act_scores)
        else:  # for act 2
            prev_node_feat = state_feat[torch.arange(len(prev_act)), prev_act, :]
            #state_feat = torch.cat(
            #    (state_feat, prev_node_feat.unsqueeze(1).expand(-1, state_feat.shape[1], -1)), dim=-1)
            #act_scores = self.act2_resnet(state_feat).squeeze(-1)
            act_query = torch.tanh(self.act2_query(prev_node_feat))
            act_scores = (act_query.unsqueeze(1) * state_feat).sum(dim=-1)
            mask = torch.zeros_like(act_scores)
            mask[torch.arange(len(prev_act)), prev_act] = -float('inf')

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


class CriticNet(torch.nn.Module):
    def __init__(
            self,
            state_feature_size,
            batch_norm,
    ):
        super(CriticNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm
        self.tensor_net = TensorNetworkModule(self.state_feature_size, self.state_feature_size)
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(self.state_feature_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, global_feat_1, global_feat_2):
        state_value = self.scoring_layer(self.tensor_net(global_feat_1, global_feat_2)).squeeze(-1)
        return state_value

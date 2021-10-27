import torch
from torch import nn
from pyg_graph_models import GCN, GraphAttentionPooling, ResNetBlock, TensorNetworkModule
from utils import construct_graph_batch, pad_tensor
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch.distributions import Categorical
from sinkhorn import Sinkhorn
from ged_ppo_bihyb_model import CriticNet


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

    def forward(self, input_graphs_1, input_graphs_2, partial_x):
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
        sim_mat = torch.bmm(node_feat_1, node_feat_2.transpose(1, 2)).detach()
        sim_mat = self.sinkhorn(sim_mat, num_nodes_1, num_nodes_2)

        partial_x = torch.stack(pad_tensor([px[:-1, :-1] for px in partial_x]))
        for b, px in enumerate(partial_x):
            graph_1_mask = px.sum(dim=-1).to(dtype=torch.bool)
            graph_2_mask = px.sum(dim=-2).to(dtype=torch.bool)
            sim_mat[b, graph_1_mask, :] = 0
            sim_mat[b, :, graph_2_mask] = 0
            sim_mat[b] = sim_mat[b] + px

        # compute cross-graph difference features
        diff_feat = node_feat_1 - torch.bmm(sim_mat, node_feat_2)

        global_feat_1 = self.att(batched_node_feat_1, batched_graphs_1.batch)
        global_feat_2 = self.att(batched_node_feat_2, batched_graphs_2.batch)

        return diff_feat, node_feat_2, global_feat_1, global_feat_2


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
        self.act2_query = nn.Linear(self.state_feature_size, self.state_feature_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_feat1, input_feat2, partial_x, known_action=None):
        return self._act(input_feat1, input_feat2, partial_x, known_action)

    def _act(self, input_feat1, input_feat2, partial_x, known_action=None):
        if known_action is None:
            known_action = (None, None)
        # roll-out 2 acts
        mask1 = self._get_mask1(partial_x)
        act1, log_prob1, entropy1 = self._select_node(input_feat1, input_feat2, mask1, known_action[0])
        mask2 = self._get_mask2(partial_x, act1)
        act2, log_prob2, entropy2 = self._select_node(input_feat1, input_feat2, mask2, known_action[1], act1)
        return torch.stack((act1, act2)), torch.stack((log_prob1, log_prob2)), entropy1 + entropy2

    def _select_node(self, node_feat1, node_feat2, mask, known_cur_act=None, prev_act=None, greedy_sel_num=0):
        node_feat1 = torch.cat((node_feat1, node_feat1.max(dim=1, keepdim=True).values), dim=1)
        # neural net prediction
        if prev_act is None:  # for act 1
            act_scores = self.act1_resnet(node_feat1).squeeze(-1)
        else:  # for act 2
            node_feat2 = torch.cat((node_feat2, node_feat2.max(dim=1, keepdim=True).values), dim=1)
            prev_node_feat = node_feat1[torch.arange(len(prev_act)), prev_act, :]
            act_query = torch.tanh(self.act2_query(prev_node_feat))
            act_scores = (act_query.unsqueeze(1) * node_feat2).sum(dim=-1)

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

    def _get_mask1(self, partial_x):
        batch_num = len(partial_x)
        act_num = max([px.shape[0] for px in partial_x])
        mask = torch.full((batch_num, act_num), -float('inf'), device=self.device)
        for b in range(batch_num):
            for available_act in (1-partial_x[b][:-1, :].sum(dim=-1)).nonzero():
                mask[b, available_act] = 0
            mask[b, -1] = 0
        return mask

    def _get_mask2(self, partial_x, prev_act):
        batch_num = len(partial_x)
        act1_num = max([px.shape[0] for px in partial_x])
        act2_num = max([px.shape[1] for px in partial_x])
        mask = torch.full((batch_num, act2_num), -float('inf'), device=self.device)
        for b in range(batch_num):
            for available_act in (1-partial_x[b][:, :-1].sum(dim=-2)).nonzero():
                mask[b, available_act] = 0
            if prev_act[b] != act1_num - 1:
                mask[b, -1] = 0
        return mask

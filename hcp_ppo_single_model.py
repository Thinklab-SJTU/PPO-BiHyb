import torch
from torch import nn
from pyg_graph_models import ResNetBlock
from torch.distributions import Categorical

from hcp_ppo_bihyb_model import GraphEncoder, CriticNet


class ActorNet(torch.nn.Module):
    def __init__(
            self,
            state_feature_size,
            batch_norm,
    ):
        super(ActorNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.act_query = nn.Linear(self.state_feature_size, self.state_feature_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_feat, node_candidates, prev_act, known_action=None):
        return self._act(input_feat, node_candidates, prev_act, known_action)

    def _act(self, input_feat, node_candidates, prev_act, known_action=None):
        mask, ready_nodes = self._get_mask(input_feat.shape[0], input_feat.shape[1], node_candidates)
        act, log_prob, entropy = self._select_node(input_feat, mask, prev_act, known_action)
        return act, log_prob, entropy

    def _select_node(self, state_feat, mask, prev_act, known_cur_act=None, greedy_sel_num=0):
        # neural net prediction
        prev_node_feat = state_feat[torch.arange(len(prev_act)), prev_act, :]
        act_query = torch.tanh(self.act_query(prev_node_feat))
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

    def _get_mask(self, batch_size, num_nodes, node_candidates):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            for node in node_candidates[b]:
                masks[b, node] = 0
                ready_nodes[b].append(node)
        return masks, ready_nodes

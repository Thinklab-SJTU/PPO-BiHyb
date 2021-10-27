from dag_ppo_bihyb_model import GraphEncoder, CriticNet
from pyg_graph_models import ResNetBlock
import torch
from torch import nn
from torch.distributions import Categorical


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

        self.act_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, state_feat, node_candidates, known_action=None):
        return self._act(state_feat, node_candidates, known_action)

    def _act(self, state_feat, node_candidates, known_action=None):
        mask1, ready_nodes1 = self._get_mask(state_feat.shape[0], state_feat.shape[1], node_candidates)
        act1, log_prob1, entropy1 = self._select_node(state_feat, mask1, known_action)
        return act1, log_prob1, entropy1

    def _select_node(self, state_feat, mask, known_cur_act=None, greedy_sel_num=0):
        # extract the void node feature (last node is the void node)
        void_node_feat = torch.max(state_feat, dim=1, keepdim=True).values
        state_feat = torch.cat((state_feat, void_node_feat), dim=1)
        act_scores = self.act_resnet(state_feat).squeeze(-1)
        mask = torch.cat((mask, torch.zeros(state_feat.shape[0], 1, device=self.device)), dim=1)
        # select action
        act_probs = nn.functional.softmax(act_scores + mask, dim=1)
        if greedy_sel_num > 0:
            argsort_prob = torch.argsort(act_probs, dim=-1, descending=True)
            acts = argsort_prob[:, :greedy_sel_num]
            acts[acts == state_feat.shape[1] - 1] = -1
            return acts, act_probs[torch.arange(acts.shape[0]).unsqueeze(-1), acts]
        else:
            dist = Categorical(probs=act_probs)
            if known_cur_act is None:
                act = dist.sample()
                act_minus1 = act.clone()
                act_minus1[act_minus1 == state_feat.shape[1] - 1] = -1
                return act_minus1, dist.log_prob(act), dist.entropy()
            else:
                known_cur_act_no_minus1 = known_cur_act.clone()
                known_cur_act_no_minus1[known_cur_act_no_minus1 == -1] = state_feat.shape[1] - 1
                return known_cur_act, dist.log_prob(known_cur_act_no_minus1), dist.entropy()

    def _get_mask(self, batch_size, num_nodes, node_candidates):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            for node in node_candidates[b]:
                masks[b, node] = 0
                ready_nodes[b].append(node)
        return masks, ready_nodes

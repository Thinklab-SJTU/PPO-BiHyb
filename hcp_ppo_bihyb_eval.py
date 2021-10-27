import time
import itertools
import torch
from copy import deepcopy
import numpy as np


def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))


def beam_search_step_kernel(idx, act_n_sel,
                            acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                            graph_list, act_list, prob_list, orig_greedy, tsp_env):
    beam_idx = idx // act_n_sel ** 2
    act1_idx = idx // act_n_sel % act_n_sel
    act2_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx].item(), probs1[beam_idx, act1_idx].item()
    act2, prob2 = acts2[beam_idx, act1_idx, act2_idx].item(), probs2[beam_idx, act1_idx, act2_idx].item()
    ready_nodes_1 = ready_nodes1[beam_idx]
    ready_nodes_2 = ready_nodes2_flat[beam_idx * act_n_sel + act1_idx]

    if act1 in ready_nodes_1 and act2 in ready_nodes_2:
        reward, new_lower_matrix, edge_candidates, new_greedy, done = \
            tsp_env.step(graph_list[beam_idx], (act1, act2), orig_greedy)
        return (
                new_lower_matrix,
                edge_candidates,
                reward,
                act_list[beam_idx] + [(act1, act2)],
                prob_list[beam_idx] + [(prob1, prob2)],
                done
        )
    else:
        return None


def beam_search(policy_model, tsp_env, inp_lower_matrix, edge_candidates, greedy_cost, max_actions, beam_size=5, multiprocess_pool=None):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    orig_greedy = greedy_cost
    best_tuple = (
        deepcopy(inp_lower_matrix),  # input lower-left adjacency matrix
        edge_candidates,  # edge candidates
        0,  # accumulated reward
        [],  # actions
        [],  # probabilities
        False,
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size

    for step in range(max_actions):
        lower_matrix_list, edge_cand_list, reward_list, act_list, prob_list = [], [], [], [], []
        for lower_matrix, edge_cand, reward, acts, probs, done in topk_graphs:
            lower_matrix_list.append(lower_matrix)
            edge_cand_list.append(edge_cand)
            reward_list.append(reward)
            act_list.append(acts)
            prob_list.append(probs)
            if done:
                return {
                    'reward': reward,
                    'solution': orig_greedy - reward,
                    'acts': acts,
                    'probs': probs,
                    'time': time.time() - start_time,
                }


        state_feat = state_encoder(lower_matrix_list)

        # mask1: (beam_size, max_num_nodes)
        mask1, ready_nodes1 = actor_net._get_mask1(state_feat.shape[0], state_feat.shape[1], edge_cand_list)
        # acts1, probs1: (beam_size, act_n_sel)
        acts1, probs1 = actor_net._select_node(state_feat, mask1, greedy_sel_num=act_n_sel)
        # acts1_flat, probs1_flat: (beam_size x act_n_sel,)
        acts1_flat, probs1_flat = acts1.reshape(-1), probs1.reshape(-1)
        # mask2_flat: (beam_size x act_n_sel, max_num_nodes)
        mask2_flat, ready_nodes2_flat = actor_net._get_mask2(
            state_feat.shape[0] * act_n_sel, state_feat.shape[1], repeat_interleave(edge_cand_list, act_n_sel),
            acts1_flat)
        # acts2_flat, probs2_flat: (beam_size x act_n_sel, act_n_sel)
        acts2_flat, probs2_flat = actor_net._select_node(
            state_feat.repeat_interleave(act_n_sel, dim=0), mask2_flat, prev_act=acts1_flat, greedy_sel_num=act_n_sel)
        # acts2, probs2: (beam_size, act_n_sel, act_n_sel)
        acts2, probs2 = acts2_flat.reshape(-1, act_n_sel, act_n_sel), probs2_flat.reshape(-1, act_n_sel, act_n_sel)

        acts1, acts2, probs1, probs2 = acts1.cpu(), acts2.cpu(), probs1.cpu(), probs2.cpu()

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                    lower_matrix_list, act_list, prob_list,
                    orig_greedy, tsp_env
                )

        if multiprocess_pool:
            pool_map = multiprocess_pool.starmap_async(
                beam_search_step_kernel, kernel_func_feeder(len(lower_matrix_list) * act_n_sel ** 2))
            tmp_graphs = pool_map.get()
        else:
            tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(lower_matrix_list) * act_n_sel ** 2)]
        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)

        # find the best action
        searched_graphs.sort(key=lambda x: x[2], reverse=True)
        if searched_graphs[0][2] > best_tuple[2]:
            best_tuple = searched_graphs[0]

        # find the topk expandable actions
        topk_graphs = searched_graphs[:beam_size]

    return {
        'reward': best_tuple[2],
        'solution': orig_greedy - best_tuple[2],
        'acts': best_tuple[3],
        'probs': best_tuple[4],
        'time': time.time() - start_time,
    }


def evaluate(policy_net, tsp_env, eval_graphs, max_steps=10, search_size=10, mp_pool=None):
    ret_result = {'reward': {}, 'optimum': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {}}
    # Load test graphs
    for graph_index, (inp_lower_matrix, edge_candidates, ori_greedy, baselines, _) in enumerate(eval_graphs):
        # Running beam search:
        bs_result = beam_search(policy_net, tsp_env, inp_lower_matrix, edge_candidates, ori_greedy, max_steps,
                                search_size, mp_pool)
        print(f'BEAMSEARCH \t'
              f'gid {graph_index} \t'
              f'time {bs_result["time"]:.2f} \t'
              f'reward {bs_result["reward"]:.4f} \t'
              f'optimum {1 if bs_result["solution"] == 0 else 0:.4f} \t'
              f'ours {bs_result["solution"]:.4f} \t'
              f'gap {bs_result["solution"] - min([v for v in baselines.values()]):.4f} \t'
              + '\t'.join([f'{key} {baselines[key]:.4f}' for key in tsp_env.available_solvers]) + '\t'
              f'action {bs_result["acts"]} \t'
              f'prob [{",".join([f"({x[0]:.3f}, {x[1]:.3f})" for x in bs_result["probs"]])}]')
        # record statistics
        ret_result['reward'][f'graph{graph_index}'] = bs_result['reward']
        ret_result['optimum'][f'graph{graph_index}'] = 1 if bs_result["solution"] == 0 else 0
        ret_result['gap'][f'graph{graph_index}'] = \
            bs_result["solution"] - min([v for v in baselines.values()])
        ret_result['solution'][f'graph{graph_index}_ours'] = bs_result["solution"]
        ret_result['num_act'][f'graph{graph_index}'] = len(bs_result["acts"])
        for key, val in baselines.items():
            ret_result['solution'][f'graph{graph_index}_{key}'] = val
        ret_result['time'][f'graph{graph_index}'] = bs_result['time']
    # compute mean
    for key, val in ret_result.items():
        if key == 'solution':
            ours_vals = []
            for sol_key, sol_val in val.items():
                if 'ours' in sol_key:
                    ours_vals.append(sol_val)
            ret_result[key]['mean'] = np.mean(ours_vals)
            ret_result[key]['std'] = np.std(ours_vals)
        else:
            ret_result[key]['mean'] = sum(val.values()) / len(val)

    print(f'BEAMSEARCH \t solution mean={ret_result["solution"]["mean"]:.4f} std={ret_result["solution"]["std"]:.4f}\t'
          f' optimum percent {ret_result["optimum"]["mean"]:.4f}')
    return ret_result


if __name__ == '__main__':
    import random
    from torch.multiprocessing import Pool, cpu_count

    from tsp_env import TSPEnv
    from hcp_ppo_bihyb_train import ActorCritic, parse_arguments

    args = parse_arguments()

    # initialize manual seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create environment
    tsp_env = TSPEnv(args.solver_type, args.min_size, args.max_size)
    args.node_feature_dim = 1

    # get current device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load training/testing data
    tuples_train, tuples_test = tsp_env.generate_tuples(args.train_sample, args.test_sample, 0)

    # init models
    ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers
    policy_net = ActorCritic(*ac_params).to(device)
    policy_net.load_state_dict(torch.load(args.test_model_weight, map_location=device))
    num_workers = cpu_count()
    mp_pool = Pool(num_workers)

    with torch.no_grad():
        evaluate(policy_net, tsp_env, tuples_test, args.max_timesteps, args.search_size, mp_pool)

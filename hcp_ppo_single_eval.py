import time
import itertools
import torch
from copy import deepcopy
import numpy as np


def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))


def beam_search_step_kernel(idx, act_n_sel,
                            acts1, probs1, ready_nodes1,
                            graph_list, act_list, prob_list, prev_sol_list, tsp_env):
    beam_idx = idx // act_n_sel
    act1_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx].item(), probs1[beam_idx, act1_idx].item()
    ready_nodes_1 = ready_nodes1[beam_idx]
    prev_sol = prev_sol_list[beam_idx]
    if len(act_list[beam_idx]) == 0:
        prev_act = 0
    else:
        prev_act = act_list[beam_idx][-1]

    if act1 in ready_nodes_1:
        reward, new_lower_matrix, node_candidates, new_sol, done = \
            tsp_env.step_e2e(graph_list[beam_idx], prev_act, act1, prev_sol)
        return (
                new_lower_matrix,
                node_candidates,
                new_sol,
                act_list[beam_idx] + [act1],
                prob_list[beam_idx] + [prob1],
                done
        )
    else:
        return None


def beam_search(policy_model, tsp_env, inp_lower_matrix, node_candidates, greedy_cost, max_actions, beam_size=5, multiprocess_pool=None):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    best_tuple = (
        deepcopy(inp_lower_matrix),  # input lower-left adjacency matrix
        node_candidates,  # edge candidates
        0,  # current tour length
        [],  # actions
        [],  # probabilities
        False,
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size

    for step in range(max_actions):
        lower_matrix_list, node_cand_list, sol_list, prev_act_list, act_list, prob_list = [], [], [], [], [], []
        for lower_matrix, node_cand, solution, acts, probs, done in topk_graphs:
            lower_matrix_list.append(lower_matrix)
            node_cand_list.append(node_cand)
            sol_list.append(solution)
            act_list.append(acts)
            if len(acts) == 0:
                prev_act_list.append(0)
            else:
                prev_act_list.append(acts[-1])
            prob_list.append(probs)
            if done:
                return {
                    'reward': -solution,
                    'solution': solution - len(inp_lower_matrix),
                    'acts': acts,
                    'probs': probs,
                    'time': time.time() - start_time,
                }


        state_feat = state_encoder(lower_matrix_list)

        # mask1: (beam_size, max_num_nodes)
        mask1, ready_nodes1 = actor_net._get_mask(state_feat.shape[0], state_feat.shape[1], node_cand_list)
        # acts1, probs1: (beam_size, act_n_sel)
        acts1, probs1 = actor_net._select_node(state_feat, mask1, torch.tensor(prev_act_list).to(actor_net.device),
                                               greedy_sel_num=act_n_sel)
        # acts1_flat, probs1_flat: (beam_size x act_n_sel,)
        acts1, probs1 = acts1.cpu(), probs1.cpu()

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, probs1, ready_nodes1,
                    lower_matrix_list, act_list, prob_list, sol_list, tsp_env
                )

        if multiprocess_pool:
            pool_map = multiprocess_pool.starmap_async(
                beam_search_step_kernel, kernel_func_feeder(len(lower_matrix_list) * act_n_sel))
            tmp_graphs = pool_map.get()
        else:
            tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(lower_matrix_list) * act_n_sel)]
        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)

        # find the topk expandable actions
        searched_graphs.sort(key=lambda x: x[2], reverse=False)
        topk_graphs = searched_graphs[:beam_size]

    print('Warning: no solution found.')
    return {
        'reward': 0,
        'solution': - 1,
        'acts': [],
        'probs': [],
        'time': time.time() - start_time,
    }


def evaluate(policy_net, tsp_env, eval_graphs, max_steps=10, search_size=10, mp_pool=None):
    ret_result = {'reward': {}, 'optimum': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {}}
    # Load test graphs
    for graph_index, (inp_lower_matrix, _, ori_greedy, baselines, _) in enumerate(eval_graphs):
        node_candidates = tsp_env.get_node_candidates(inp_lower_matrix, len(inp_lower_matrix))
        # Running beam search:
        bs_result = beam_search(policy_net, tsp_env, inp_lower_matrix, node_candidates, ori_greedy, max_steps,
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
              f'prob [{",".join([f"{x:.3f}" for x in bs_result["probs"]])}]')
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

    print(f'BEAMSEARCH \t solution mean={ret_result["solution"]["mean"]:.4f} std={ret_result["solution"]["std"]:.4f} \t'
          f' optimum percent {ret_result["optimum"]["mean"]:.4f}')
    return ret_result


if __name__ == '__main__':
    import random
    from torch.multiprocessing import Pool, cpu_count

    from tsp_env import TSPEnv
    from hcp_ppo_single_train import ActorCritic, parse_arguments

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

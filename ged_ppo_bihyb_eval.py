import time
import itertools
import torch
import numpy as np


def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))


def beam_search_step_kernel(idx, act_n_sel,
                            acts1, acts2, probs1, probs2,
                            graph_1_list, graph_2_list, ori_k_list,
                            act_list, prob_list, orig_greedy, ged_env):
    beam_idx = idx // act_n_sel ** 2
    act1_idx = idx // act_n_sel % act_n_sel
    act2_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx], probs1[beam_idx, act1_idx]
    act2, prob2 = acts2[beam_idx, act1_idx, act2_idx], probs2[beam_idx, act1_idx, act2_idx]
    acts = torch.stack((act1, act2))

    if prob1 > 1e-3 and prob2 > 1e-3:
        reward, new_graph_1, new_greedy = \
            ged_env.step(graph_1_list[beam_idx], graph_2_list[beam_idx], ori_k_list[beam_idx], acts, orig_greedy)
        return (
                new_graph_1,
                graph_2_list[beam_idx],
                ori_k_list[beam_idx],
                reward,
                act_list[beam_idx] + [(act1.item(), act2.item())],
                prob_list[beam_idx] + [(prob1.item(), prob2.item())],
        )
    else:
        return None


def beam_search(policy_model, ged_env, inp_graph_1, inp_graph_2, ori_ks, greedy_cost, max_actions, beam_size=5, multiprocess_pool=None):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    graph_1_copy = inp_graph_1.clone()
    orig_greedy = greedy_cost
    best_tuple = (
        graph_1_copy,  # graph 1
        inp_graph_2,  # graph 2
        ori_ks,  # original k
        torch.Tensor([0]).to(ori_ks.device),  # accumulated reward
        [],  # actions
        [],  # probabilities
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size

    for step in range(max_actions):
        searched_graphs = []
        graph_1_list, graph_2_list, ori_k_list, reward_list, act_list, prob_list = [], [], [], [], [], []
        for graph_1, graph_2, ori_k, reward, acts, probs in topk_graphs:
            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)
            ori_k_list.append(ori_k)
            reward_list.append(reward)
            act_list.append(acts)
            prob_list.append(probs)

        diff_feat, global_feat_1, global_feat_2 = state_encoder(graph_1_list, graph_2_list)

        # acts1, probs1: (beam_size, act_n_sel)
        acts1, probs1 = actor_net._select_node(diff_feat, greedy_sel_num=act_n_sel)
        # acts1_flat, probs1_flat: (beam_size x act_n_sel,)
        acts1_flat, probs1_flat = acts1.reshape(-1), probs1.reshape(-1)
        # acts2_flat, probs2_flat: (beam_size x act_n_sel, act_n_sel)
        acts2_flat, probs2_flat = actor_net._select_node(
            diff_feat.repeat_interleave(act_n_sel, dim=0), prev_act=acts1_flat, greedy_sel_num=act_n_sel)
        # acts2, probs2: (beam_size, act_n_sel, act_n_sel)
        acts2, probs2 = acts2_flat.reshape(-1, act_n_sel, act_n_sel), probs2_flat.reshape(-1, act_n_sel, act_n_sel)

        #acts1, acts2, probs1, probs2 = acts1.cpu(), acts2.cpu(), probs1.cpu(), probs2.cpu()
        #cpu_device = torch.device('cpu')
        #graph_1_list = [_.to(cpu_device) for _ in graph_1_list]
        #graph_2_list = [_.to(cpu_device) for _ in graph_2_list]
        #ori_k_list = [_.cpu() for _ in ori_k_list]

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, acts2, probs1, probs2,
                    graph_1_list, graph_2_list, ori_k_list,
                    act_list, prob_list, orig_greedy, ged_env
                )

        if multiprocess_pool:
            pool_map = multiprocess_pool.starmap_async(
                beam_search_step_kernel, kernel_func_feeder(len(graph_1_list) * act_n_sel ** 2))
            tmp_graphs = pool_map.get()
        else:
            tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(graph_1_list) * act_n_sel ** 2)]
        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)

        # find the best action
        searched_graphs.sort(key=lambda x: x[3], reverse=True)
        if searched_graphs[0][3] > best_tuple[3]:
            best_tuple = searched_graphs[0]

        # find the topk expandable actions
        topk_graphs = searched_graphs[:beam_size]

    return {
        'reward': best_tuple[3].item(),
        'solution': (orig_greedy - best_tuple[3]).item(),
        'acts': best_tuple[4],
        'probs': best_tuple[5],
        'time': time.time() - start_time,
    }


def evaluate(policy_net, ged_env, eval_graphs, max_steps=10, search_size=10, mp_pool=None):
    ret_result = {'reward': {}, 'ratio': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {}}
    # Load test graphs
    for graph_index, (inp_graph_1, inp_graph_2, ori_k, ori_greedy, baselines, _) in enumerate(eval_graphs):
        # Running beam search:
        bs_result = beam_search(policy_net, ged_env, inp_graph_1, inp_graph_2, ori_k, ori_greedy, max_steps, search_size, mp_pool)
        print(f'BEAMSEARCH \t'
              f'gid {graph_index} \t'
              f'time {bs_result["time"]:.2f} \t'
              f'reward {bs_result["reward"]:.4f} \t'
              f'ratio {bs_result["reward"] / ori_greedy:.4f} \t'
              f'ours {bs_result["solution"]:.4f} \t'
              f'gap {(bs_result["solution"] - min([v for v in baselines.values()])) / bs_result["solution"]:.4f} \t'
              + '\t'.join([f'{key} {baselines[key]:.4f}' for key in ged_env.available_solvers]) + '\t'
              f'action {bs_result["acts"]} \t'
              f'prob [{",".join([f"({x[0]:.3f}, {x[1]:.3f})" for x in bs_result["probs"]])}]')
        # record statistics
        ret_result['reward'][f'graph{graph_index}'] = bs_result['reward']
        ret_result['ratio'][f'graph{graph_index}'] = bs_result["reward"] / ori_greedy
        ret_result['gap'][f'graph{graph_index}'] = \
            (bs_result["solution"] - min([v for v in baselines.values()])) / bs_result["solution"]
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
          f' mean ratio {ret_result["ratio"]["mean"]:.4f}')
    return ret_result


if __name__ == '__main__':
    import random
    from torch.multiprocessing import Pool, cpu_count

    from utils.ged_env import GEDenv
    from ged_ppo_bihyb_train import ActorCritic, parse_arguments

    args = parse_arguments()

    # initialize manual seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create environment
    ged_env = GEDenv(args.solver_type, args.dataset)
    args.node_feature_dim = ged_env.feature_dim

    # get current device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load training/testing data
    tuples_train = ged_env.generate_tuples(ged_env.training_graphs, 50, 0, device)
    tuples_test = ged_env.generate_tuples(ged_env.val_graphs, 10, 1, device)

    # init models
    ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers
    policy_net = ActorCritic(*ac_params).to(device)
    policy_net.load_state_dict(torch.load(args.test_model_weight, map_location=device))
    num_workers = cpu_count()
    mp_pool = Pool(num_workers)

    with torch.no_grad():
        evaluate(policy_net, ged_env, tuples_test, args.max_timesteps, args.search_size,
                 None if torch.cuda.is_available() else mp_pool)

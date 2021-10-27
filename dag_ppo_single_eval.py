import time
import itertools
import torch
import numpy as np


def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))


def beam_search_step_kernel(idx, act_n_sel,
                            acts1, probs1, ready_nodes1,
                            graph_list, act_list, prob_list, prev_makespan, dag_model):
    beam_idx = idx // act_n_sel
    act1_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx].item(), probs1[beam_idx, act1_idx].item()
    if act1 in ready_nodes1[beam_idx] + [-1]:
        assert prob1 > 0
        reward, new_graph, new_makespan, node_candidates, done = \
            dag_model.step_e2e(graph_list[beam_idx], act1, prev_makespan[beam_idx])
        return (
                new_graph,
                new_makespan,
                act_list[beam_idx] + [act1],
                prob_list[beam_idx] + [prob1],
                node_candidates,
                done
        )
    else:
        return None


def beam_search(policy_model, dag_model, inp_graph, max_actions, beam_size=5, multiprocess_pool=None):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    graph_copy = inp_graph.copy()
    best_tuple = (
        graph_copy,  # graph
        0,  # accumulated makespan
        [],  # actions
        [],  # probabilities
        dag_model.get_node_candidates(graph_copy),  # edge candidates
        False,  # stop flag
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size

    for step in range(max_actions):
        graph_list, makespan_list, act_list, prob_list, node_cand_list = [], [], [], [], []
        for graph, makespan, acts, probs, node_cand, done in topk_graphs:
            assert done is False
            graph_list.append(graph)
            makespan_list.append(makespan)
            act_list.append(acts)
            prob_list.append(probs)
            node_cand_list.append(node_cand)

        state_feat = state_encoder(graph_list)

        # mask1: (beam_size, max_num_nodes)
        mask1, ready_nodes1 = actor_net._get_mask(state_feat.shape[0], state_feat.shape[1], node_cand_list)
        # acts1, probs1: (beam_size, act_n_sel)
        acts1, probs1 = actor_net._select_node(state_feat, mask1, greedy_sel_num=act_n_sel)

        acts1, probs1 = acts1.cpu(), probs1.cpu()

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, probs1, ready_nodes1,
                    graph_list, act_list, prob_list,
                    makespan_list, dag_model
                )

        if multiprocess_pool:
            pool_map = multiprocess_pool.starmap_async(
                beam_search_step_kernel, kernel_func_feeder(len(graph_list) * act_n_sel))
            tmp_graphs = pool_map.get()
        else:
            tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(graph_list) * act_n_sel)]
        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)

        # find the topk expandable actions
        searched_graphs.sort(key=lambda x: x[1], reverse=False)
        topk_graphs = []
        for g in searched_graphs[:beam_size]:
            if g[5]:
                best_tuple = g
                break
            else:
                topk_graphs.append(g)
        if best_tuple[1] != 0:
            break

    return {
        'reward': -best_tuple[1],
        'solution': best_tuple[1],
        'acts': best_tuple[2],
        'probs': best_tuple[3],
        'time': time.time() - start_time,
    }


def evaluate(policy_net, dag_graph, eval_graphs, max_steps=10, search_size=10, mp_pool=None):
    ret_result = {'reward': {}, 'ratio': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {}}
    # Load test graphs
    for graph_index, (inp_graph, _, _, baselines) in enumerate(eval_graphs):
        # Running beam search:
        bs_result = beam_search(policy_net, dag_graph, inp_graph, max_steps, search_size, mp_pool)
        print(f'BEAMSEARCH \t'
              f'gid {graph_index} \t'
              f'time {bs_result["time"]:.2f} \t'
              f'reward {bs_result["reward"]:.4f} \t'
              f'ours {bs_result["solution"]:.4f} \t'
              f'gap {(bs_result["solution"] - min([v for v in baselines.values()])) / bs_result["solution"]:.4f} \t'
              f'sfs {baselines["shortest_first"]:.4f} \t'
              f'cp {baselines["critical_path"]:.4f} \t'
              f'ts {baselines["tetris"]:.4f} \t'
              f'action {bs_result["acts"]} \t'
              f'prob {",".join([f"{x:.3f}" for x in bs_result["probs"]])}')
        # record statistics
        ret_result['reward'][f'graph{graph_index}'] = bs_result['reward']
        best_baseline = min([v for v in baselines.values()])
        ret_result['ratio'][f'graph{graph_index}'] = (best_baseline - bs_result["solution"]) / best_baseline
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

    from dag_graph import DAGraph
    from dag_data.dag_generator import load_tpch_tuples
    from dag_ppo_single_train import ActorCritic, parse_arguments

    args = parse_arguments()

    # initialize manual seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create DAG graph environment
    resource_dim = 1
    raw_node_feature_dim = 1 + resource_dim  # (duration, resources)
    args.node_feature_dim = raw_node_feature_dim
    dag_graph = DAGraph(resource_dim=resource_dim,
                        feature_dim=args.node_feature_dim,
                        scheduler_type=args.scheduler_type)

    # load training/testing data
    vargs = (
        dag_graph,
        args.num_init_dags,
        raw_node_feature_dim,
        resource_dim,
        args.resource_limit,
        args.add_graph_features,
        args.scheduler_type
    )
    tuples_train, tuples_test = \
        load_tpch_tuples(args.train_sample, 0, *vargs), load_tpch_tuples(args.test_sample, 1, *vargs)

    # get current device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init models
    ac_params = dag_graph, args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, \
                args.gnn_layers
    policy_net = ActorCritic(*ac_params).to(device)
    policy_net.load_state_dict(torch.load(args.test_model_weight, map_location=device))
    num_workers = cpu_count()
    mp_pool = Pool(num_workers)

    with torch.no_grad():
        evaluate(policy_net, dag_graph, tuples_test, args.max_timesteps, args.search_size, mp_pool)

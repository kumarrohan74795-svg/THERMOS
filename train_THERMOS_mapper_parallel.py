import argparse
import os
import random
import time
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import simpy
import matplotlib.pyplot as plt                                                 
import numpy as np
from  morl import common, RL_variables, chiplets_RL_env
from chiplet_system_simulator import ChipletSystem

import itertools
from datetime import datetime
import torch
import torch.multiprocessing as mp
import csv

from torch.utils.tensorboard import SummaryWriter
from morl.agents import MOPPOAgent
from morl.networks import MOPPO_Actor_Critic, FDDTN
import copy
from collections import namedtuple

def compute_sparsity(obj_batch):
    non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)        
    objs = obj_batch[non_dom]
    sparsity_sum = 0     
    for objective in range(objs.shape[-1]):
        objs_sort = np.sort(objs[:,objective])
        sp = 0
        for i in range(len(objs_sort)-1):
            sp +=  np.power(objs_sort[i] - objs_sort[i+1],2)
        sparsity_sum += sp
    if len(objs) > 1:
        sparsity = sparsity_sum/(len(objs)-1)
    else:   
        sparsity = 0
    
    return sparsity

def global_reset_eval(args, evaluation, wl_file=None, eval_inj_rate = 0.01):
    ## Initialize Simpy environment
    simpy_env = simpy.Environment(initial_time = 0)

    ## Create an instance of the RL information class to share among the different modules
    common.rl_state_info = common.RLStateInfo()

    ## Add synchronization events
    common.rl_state_info.wait_for_task   = simpy_env.event()
    common.rl_state_info.wait_for_action = simpy_env.event()
    common.rl_state_info.wait_for_sched  = simpy_env.event()
    common.rl_state_info.wait_for_ptask  = simpy_env.event()
    common.rl_state_info.sim_done        = simpy_env.event()

    ## Initialize ChipletSystem simulator object
    if evaluation:
        chiplet_sim = ChipletSystem(simpy_env=simpy_env,
                                    chiplet_mapping_file=args.chiplet_mapping_file,
                                    adj_matrix_file=args.adj_matrix_file,
                                    inj_rate=eval_inj_rate,
                                    mapper='rl',
                                    test=True,
                                    wl_file=wl_file)
    else:
        chiplet_sim = None

    ## Create a chiplets RL environment
    chiplet_env_for_RL = chiplets_RL_env.Chiplets_rl_env(simpy_env=simpy_env, 
                                                         args=args)
    
    if evaluation:
        simpy_env.process(chiplet_sim.execute_simulation_rl_disc_clustered())
    return simpy_env, chiplet_sim, chiplet_env_for_RL

Queue_obj = namedtuple('Queue_obj', ['batch_values', 'time_step','ep_cnt','process_ID'])

def child_process(actor, critic, args, train_queue, p_id, w_batch):
    ## Global reset in the child process
    def global_reset(args, evaluation, start_sim = True, inj_rate = 0.01):
        ## Initialize Simpy environment
        simpy_env = simpy.Environment(initial_time = 0)

        ## Create an instance of the RL information class to share among the different modules
        common.rl_state_info = common.RLStateInfo()

        ## Add synchronization events
        common.rl_state_info.wait_for_task   = simpy_env.event()
        common.rl_state_info.wait_for_action = simpy_env.event()
        common.rl_state_info.wait_for_sched  = simpy_env.event()
        common.rl_state_info.wait_for_ptask  = simpy_env.event()
        common.rl_state_info.sim_done        = simpy_env.event()

        ## Initialize ChipletSystem simulator object
        if evaluation:
            chiplet_sim = None
        else:
            chiplet_sim = ChipletSystem(simpy_env=simpy_env,
                                        chiplet_mapping_file=args.chiplet_mapping_file,
                                        adj_matrix_file=args.adj_matrix_file,
                                        inj_rate=inj_rate,
                                        mapper='rl',
                                        total_jobs=100,
                                        test=False)

        chiplet_env_for_RL = chiplets_RL_env.Chiplets_rl_env(simpy_env=simpy_env, 
                                                             args=args)
        
        if start_sim:
            simpy_env.process(chiplet_sim.execute_simulation_rl_disc_clustered())
        return simpy_env, chiplet_sim, chiplet_env_for_RL

    def run_episode(chiplet_env_for_RL, simpy_env, args, episode_queue, p_id) :  

        rl_variables.next_obs = torch.Tensor(chiplet_env_for_RL.reset()).to(args.device)
        rl_variables.next_done = torch.zeros(1).to(args.device)
        rl_variables.agent.w_ep = None
        done = False 
        step_cnt = 0
        while not done :
            
            rl_variables.dones[rl_variables.batch_step] = rl_variables.next_done

            if common.DEBUG_RL :
                print(f'[ INFO] [RUN] Waiting for task...')
 
            yield common.rl_state_info.wait_for_task
            common.rl_state_info.wait_for_task = simpy_env.event()
 
            if common.DEBUG_RL :
                print('[ INFO] [RUN] Done with waiting for task, received task ..' )
 
            next_obs = copy.deepcopy(common.rl_state_info.current_state)
            rl_variables.next_obs = torch.Tensor(next_obs).to(args.device)
            rl_variables.obs[rl_variables.batch_step] = rl_variables.next_obs

            with torch.no_grad():
                action, logprob, _, value, valid_action = rl_variables.agent.get_action_and_value(rl_variables.next_obs, mask=common.rl_state_info.supported_clusters)

                rl_variables.values[rl_variables.batch_step] = value
            rl_variables.actions[rl_variables.batch_step] = action
            rl_variables.logprobs[rl_variables.batch_step] = logprob
            rl_variables.preferences[rl_variables.batch_step] = rl_variables.agent.w_ep
            rl_variables.action_masks[rl_variables.batch_step] = valid_action

            common.rl_state_info.action = action.cpu().numpy()
            
            ## Perform an action in the environment
            yield simpy_env.process(chiplet_env_for_RL.step(common.rl_state_info.action))

            reward = common.rl_state_info.reward
            terminal = common.rl_state_info.terminal 
            done = common.rl_state_info.done

            rl_variables.rewards[rl_variables.batch_step] = torch.tensor(reward).to(args.device).view(-1)
            rl_variables.next_done = torch.tensor(int(done)).to(args.device)

            if common.DEBUG_RL:
                print(f"Step: {step_cnt}, Reward: {reward}, terminal: {terminal}, done: {done}")

            step_cnt += 1
            if done:
                rl_variables.episode_cnt += 1
                next_obs = copy.deepcopy(common.rl_state_info.current_state)
                rl_variables.next_obs = torch.Tensor(next_obs).to(args.device)
                step_cnt = 0
                rl_variables.train_w = copy.deepcopy(rl_variables.agent.w_ep)
                rl_variables.agent.w_ep = None
                
            rl_variables.global_steps += 1
            rl_variables.batch_step += 1
            
            if common.DEBUG_RL:
                print(f'[debug][info] {rl_variables.batch_step}, p_id {p_id}')

            if rl_variables.batch_step == args.num_steps:
                rl_variables.batch_step = 0

                with torch.no_grad():
                    next_value = rl_variables.agent.get_value(rl_variables.next_obs.unsqueeze(0), rl_variables.preferences[-1].unsqueeze(0))
                    if args.gae:
                        advantages = torch.zeros_like(rl_variables.rewards).to(args.device)
                        lastgaelam = 0
                        for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - rl_variables.next_done
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - rl_variables.dones[t + 1]
                                nextvalues = rl_variables.values[t + 1]
                            delta = rl_variables.rewards[t] + args.gamma * nextvalues * nextnonterminal - rl_variables.values[t]
                            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                            advantages[t] = lastgaelam
                        returns = advantages + rl_variables.values
                    else:
                        returns = torch.zeros_like(rl_variables.rewards).to(args.device)
                        for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - rl_variables.next_done
                                next_return = next_value
                            else:
                                nextnonterminal = 1.0 - rl_variables.dones[t + 1]
                                next_return = returns[t + 1]
                            returns[t] = rl_variables.rewards[t] + args.gamma * nextnonterminal * next_return
                        advantages = returns - rl_variables.values

                # Batch Update 
                b_obs = rl_variables.obs
                b_logprobs = rl_variables.logprobs
                b_actions = rl_variables.actions
                b_advantages = advantages
                b_returns = returns
                b_values = rl_variables.values
                b_preferences = rl_variables.preferences
                b_masks = rl_variables.action_masks
                batch_values = [b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_preferences, b_masks]

                rl_variables.batch_values = batch_values

                episode_queue.put(Queue_obj(batch_values = copy.deepcopy(rl_variables.batch_values), time_step = rl_variables.global_steps, ep_cnt = rl_variables.episode_cnt, process_ID = p_id))

            ## Debug statement
            if common.DEBUG_RL :
                print(f'[ INFO] Step complete for with action {common.rl_state_info.action} and reward {common.rl_state_info.reward}')

            ## Handshake for current task process completion
            common.rl_state_info.wait_for_ptask.succeed()
    
        ## while not done :
        common.rl_state_info.sim_done.succeed()
        return
    ## def run_episode() :
    
    torch.set_num_threads(1)
    torch.manual_seed(args.seed + 10*p_id)
    random.seed(args.seed + 10*p_id)
    np.random.seed(args.seed + 10*p_id)
    os.environ['PYTHONHASHSEED'] = str(args.seed + 10*p_id)
    torch.backends.cudnn.deterministic = True

    rl_variables = RL_variables.RLVariables()

    args.p_id = p_id
   
    injection_rate_index = random.choices(args.injection_rate, k=1000)
    simpy_env, chiplet_sim, chiplet_env_for_RL = global_reset(args, evaluation = False, start_sim = False)

    agent = MOPPOAgent(actor,critic,args).to(args.device)
    # Key Preference Selection
    agent.w_batch = w_batch
    
    obs = torch.zeros((args.num_steps, args.obs_shape)).to(args.device)
    actions = torch.zeros((args.num_steps,) + chiplet_env_for_RL.action_space.shape).to(args.device)
    action_masks = torch.zeros((args.num_steps, chiplet_env_for_RL.action_space.n)).to(args.device)

    logprobs = torch.zeros((args.num_steps)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.reward_size)).to(args.device)
    dones = torch.zeros((args.num_steps )).to(args.device)
    values = torch.zeros((args.num_steps, args.reward_size)).to(args.device)
    preferences = torch.zeros((args.num_steps, args.reward_size)).to(args.device)

    rl_variables.agent = agent
    rl_variables.obs = obs
    rl_variables.actions = actions
    rl_variables.logprobs = logprobs
    rl_variables.rewards = rewards
    rl_variables.dones = dones
    rl_variables.values = values
    rl_variables.preferences = preferences
    rl_variables.action_masks = action_masks

    iteration = 0

    while True:
        common.iteration = iteration
        common.rl_seed = common.iteration

        inj_rate = injection_rate_index[0]
        injection_rate_index.append(injection_rate_index.pop(0))

        simpy_env, chiplet_sim, chiplet_env_for_RL = global_reset(args, evaluation = False, inj_rate=inj_rate)

        ## For first task, preemptively trigger event so that simulation doesn't hang
        common.rl_state_info.wait_for_ptask.succeed()

        simpy_env.process(run_episode(chiplet_env_for_RL, simpy_env, args,train_queue, p_id))
        simpy_env.run(until = common.rl_state_info.sim_done)
        iteration += 1
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate_critic", type=float, default=5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--learning_rate_actor", type=float, default=5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--injection_rate", type=float, default=0.01, help="injection rate")
    parser.add_argument("--eval_injection_rate", type=float, default=0.01,  help="injection rate")
    parser.add_argument("--hidden_size", type=int, default = 64, help="number of neurons in a layer")
    parser.add_argument("--seed", type=int, default = 1, help="seed of the experiment")
    parser.add_argument("--total_timesteps", type=int, default=80000000, help="total timesteps of the experiments")
    parser.add_argument("--torch_deterministic", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--load_model", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, nargs="?", const=False, help="load previously trained model")

    # Chiplet system specific arguments
    parser.add_argument("--n_chiplets", type=int, default=78, help="number of chiplets in the system")
    parser.add_argument("--n_clusters", type=int, default=4, help="number of clusters in the system")
    parser.add_argument("--adj_matrix_file", type=str, default='params/adj_matrix_86x86.csv', help="adjacency matrix file")
    parser.add_argument("--chiplet_mapping_file", type=str, default='params/chiplet_mapping_78.yaml', help="chiplet mapping file")

    # DT specific arguments 
    parser.add_argument("--tree_depth", type=int, default=6, help="depth of the decision tree")
    parser.add_argument("--tree_alpha", type=int, default=5, help="alpha value used in DT")

    # Algorithm specific arguments
    parser.add_argument("--num_envs", type=int, default=3,help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default = 10240, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gae", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, nargs="?", const=True, help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.95, help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.99, help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default = 176, help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default = 30, help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default = 0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default = 0.00, help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default = 1, help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default = 3, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default = None, help="the target KL divergence threshold")

    args = parser.parse_args()

    args.injection_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    args.eval_injection_rate = [0.1, 0.25, 0.4]
    
    args.batch_size = int(args.num_envs * args.num_steps) 
    args.minibatch_size = int(args.batch_size // args.num_minibatches) 
    args.evaluation_episode = 20
    return args

def run_episode_eval(chiplet_env_for_RL, simpy_env, preference_eval, agent, args) :  
    preference_eval = torch.FloatTensor(preference_eval)
    next_obs = torch.Tensor(chiplet_env_for_RL.reset()).to(args.device)
    episode_reward = 0
    done = False
    iteration = 0
    while not done :
        if common.DEBUG_RL :
            print('[ INFO] [RUN] [eval] Waiting for task...')

        yield common.rl_state_info.wait_for_task
        common.rl_state_info.wait_for_task = simpy_env.event()

        if common.DEBUG_RL :
            print('[ INFO] [RUN] [eval] Done with waiting for task, received task ...')

        next_obs = copy.deepcopy(common.rl_state_info.current_state)
        next_obs = torch.Tensor(next_obs).to(args.device)

        with torch.no_grad():
            action, _ = agent.get_action_and_value(next_obs.unsqueeze(0), preference=preference_eval.unsqueeze(0), deterministic = True, mask=common.rl_state_info.supported_clusters)

        common.rl_state_info.action = action

        ## Perform an action in the environment
        yield simpy_env.process(chiplet_env_for_RL.step(common.rl_state_info.action))
        reward      = common.rl_state_info.reward
        done        = common.rl_state_info.done

        if common.DEBUG_RL :
            print(f'[ INFO] [eval] Step complete for task with action {common.rl_state_info.action} and reward {common.rl_state_info.reward}')

        ## Handshake for current task process completion
        common.rl_state_info.wait_for_ptask.succeed()
        
        episode_reward += reward
        iteration += 1
        common.results.episode_return = episode_reward
        common.results.episode_iterations = iteration
    agent.w_ep = None
    common.rl_state_info.sim_done.succeed()
    return


def generate_w_batch_test(args, step_size):
    mesh_array = []
    step_size = step_size
    for i in range(args.reward_size):
        mesh_array.append(np.arange(0,1+step_size, step_size))
        
    w_batch_test = np.array(list(itertools.product(*mesh_array)))
    w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1,:]
    w_batch_test = np.unique(w_batch_test,axis =0)

    return w_batch_test

if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(1)
    mp.set_start_method('spawn', force=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    network = 'mix_6wl'
    method = f'dt_cluster_mo_{network}_latency_energy'

    simpy_env, chiplet_sim, chiplet_env_for_RL = global_reset_eval(args,  evaluation = False)
    name_env = "Chiplets-RL"

    print(f"initial reset done \n")

    log_dir = f'runs/{datetime.now().strftime("%d_%b_%Y_%H_%M")}THERMOS_{name_env}_{args.eval_injection_rate}_{method}'
    writer = SummaryWriter(log_dir, comment = ' THERMOS ' + name_env)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cpu")
    args.device = device

    #Initialize environment related arguments
    args.obs_shape = chiplet_env_for_RL.observation_space.shape[0]
    args.action_shape = chiplet_env_for_RL.action_space.n
    args.reward_size = 2

    model = MOPPO_Actor_Critic(args, FDDTN(args))

    actor = model.actor
    critic = model.critic

    actor.share_memory()
    critic.share_memory()

    agent = MOPPOAgent(actor, critic, args).to(args.device)

    args.w_step_size = 0.5
    w_batch_test = generate_w_batch_test(args,step_size = args.w_step_size)    
    print(f"Preference Space: {w_batch_test}")

    agent.w_batch = w_batch_test

    args.load_model = False
    if args.load_model:
        agent_actor = torch.load(f'eval_models/{method}/best_actor_3.93_41.pkl')
        agent_critic = torch.load(f'eval_models/{method}/best_critic_3.93_41.pkl')
        agent.actor.load_state_dict(agent_actor)
        agent.critic.load_state_dict(agent_critic)

    start_time = time.time()
    global_steps = 0
    args.num_updates = args.total_timesteps // args.batch_size

    w_batch_test_split = np.array_split(w_batch_test,args.num_envs)

    # Initialize child processes
    train_queue_list = []
    data_proc_list = []
    for p_id in range(args.num_envs):
        train_queue = mp.JoinableQueue(maxsize=1)
        data_proc = mp.Process(target=child_process,
                               args=(actor, critic, args, train_queue, p_id, w_batch_test_split[p_id]))
        data_proc.start()
        train_queue_list.append(train_queue)
        data_proc_list.append(data_proc)

    iteration = 0
    eval_cnt = 0
    args.ppo_update = 0
    global_steps = 0
    start_time = time.time()
    process_step_array = np.zeros((args.num_envs,))
    process_episode_array = np.zeros((args.num_envs,))
    
    # if eval_inj_rate is a single value and not a list
    if not isinstance(args.eval_injection_rate, list):
        args.eval_injection_rate = [args.eval_injection_rate]

    if not os.path.exists(f'rl_results/eval/{method}/'):
        os.makedirs(f'rl_results/eval/{method}/')

    # for update in range(1, args.num_updates + 1):
    print(f"Starting PPO Updates")
    try:
        while args.ppo_update <= args.num_updates + 1:

            evaluation_episode = args.ppo_update % args.evaluation_episode == 0
            batch_values_main = []

            m_obs, m_logprobs, m_actions, m_advantages, m_returns, m_values, m_preferences, m_masks = [], [], [], [], [], [], [], []

            for main_cnt in range(args.num_envs):
                process_samples = train_queue_list[main_cnt].get()
                process_step_array[process_samples.process_ID] = process_samples.time_step
                process_episode_array[process_samples.process_ID] = process_samples.ep_cnt

                b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_preferences, b_masks = process_samples.batch_values

                m_obs.append(b_obs)
                m_logprobs.append(b_logprobs)
                m_actions.append(b_actions)
                m_advantages.append(b_advantages)
                m_returns.append(b_returns)
                m_values.append(b_values)
                m_preferences.append(b_preferences)
                m_masks.append(b_masks)

            batch_values_main = [ torch.cat(m_obs), torch.cat(m_logprobs), torch.cat(m_actions), torch.cat(m_advantages), torch.cat(m_returns), torch.cat(m_values), torch.cat(m_preferences), torch.cat(m_masks)]
        
            # Update networks           
            agent.learn(writer,sum(process_step_array), batch_values_main)
            args.ppo_update +=1
            print(f"PPO Update Epoch: {args.ppo_update}, Time steps of Each Process: {process_step_array}, Episode Count of Each Process: {process_episode_array}")
            # Reading from Queue is done
            for main_cnt in range(args.num_envs):
                train_queue_list[main_cnt].task_done()

            iteration += 1
            if evaluation_episode :
                print("")
                print(f"[I] Starting Evaluation Iteration {args.ppo_update}")
                print(f"[I] Global Time Step {sum(process_step_array)}")
                eval_objs = np.zeros((w_batch_test.shape))
                for inj_rate in args.eval_injection_rate:
                    pref_cnt = 0
                    for preference_eval in w_batch_test:
                        eval_cnt = 0
                        job_exe_time_eval= []
                        job_energy_consumption_eval= []

                        workloads = [f'{network}_200_1000-20000.csv']

                        for workload in workloads:
                            common.rl_seed = eval_cnt

                            workload_file = f'workloads/{workload}'
                            simpy_env, chiplet_sim, chiplet_env_for_RL = global_reset_eval(args, evaluation = True, wl_file=workload_file, eval_inj_rate=inj_rate)

                            ## For first task, preemptively trigger event so that simulation doesn't hang
                            common.rl_state_info.wait_for_ptask.succeed()

                            simpy_env.process(run_episode_eval(chiplet_env_for_RL, simpy_env, preference_eval, agent, args))

                            simpy_env.run(until = common.rl_state_info.sim_done)

                            ## Print statistics at the end of the simulation
                            if common.results.completed_jobs > 0 :
                                # update these values
                                completion_rate_jobs = (common.results.completed_jobs) / (common.results.simulation_time)
                                job_exe_time = common.results.average_execution_time
                                energy_consumption = common.results.average_energy_consumption
                            else :
                                completion_rate_jobs = 0
                                job_exe_time = 0
                                energy_consumption = 0

                            eval_cnt += 1

                            job_exe_time_eval.append(job_exe_time)
                            job_energy_consumption_eval.append(energy_consumption)

                            wl_name = workload.split('.')[0]
                            with open(f'rl_results/eval/{method}/eval_{wl_name}_{inj_rate}_{method}_results.csv', mode='a') as file:
                                file_w = csv.writer(file)
                                # if first time, write the header
                                if file.tell() == 0:
                                    file_w.writerow(['Episode', 'preference', 'completion_rate (jobs/s)', 
                                                    'total_neural_nets', 'average_execution_time (s)', 
                                                    'average_energy_consumed (J)', 'Return', 'Iterations'])
                                file_w.writerow([args.ppo_update, preference_eval, completion_rate_jobs,
                                                 common.results.completed_jobs, common.results.average_execution_time, 
                                                common.results.average_energy_consumption, common.results.episode_return, common.results.episode_iterations])

                        eval_job_exe_time = np.array(job_exe_time_eval).mean()
                        eval_energy_consumption = np.array(job_energy_consumption_eval).mean()

                        # for last value in eval_injection_rate
                        if inj_rate == args.eval_injection_rate[-2]:
                            eval_objs[pref_cnt,:] = np.array([eval_job_exe_time,eval_energy_consumption])

                        print(f'[I] Completed iteration with preference {preference_eval}: avg execution time:{eval_job_exe_time}, energy consumption:{eval_energy_consumption}')

                        pref_cnt +=1

                # Compute hypervolume and sparsity
                perf_ind = HV(ref_point = np.zeros((eval_objs.shape[1])))
                hypervolume = perf_ind(-eval_objs)
                sparsity = compute_sparsity(-eval_objs)

                eval_metric = hypervolume
                
                writer.add_scalar('Results/Hypervolume'.format(), hypervolume, args.ppo_update)
                writer.add_scalar('Results/Sparsity'.format(), sparsity, args.ppo_update)
                # Save Best Model
                
                # check if directory exists
                if not os.path.exists(f"eval_models/{method}"):
                    os.makedirs(f"eval_models/{method}")

                torch.save(agent.actor.state_dict(), f"eval_models/{method}/best_actor_{round(hypervolume,2)}_{args.ppo_update}.pkl")
                torch.save(agent.critic.state_dict(), f"eval_models/{method}/best_critic_{round(hypervolume,2)}_{args.ppo_update}.pkl")

        print(f"---Training lasts {(time.time() - start_time)/60} minutes ---")
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()  

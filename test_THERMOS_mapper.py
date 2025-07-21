import argparse
import os
import random
import time

import simpy
import numpy as np
from  morl import common, chiplets_RL_env
from chiplet_system_simulator import ChipletSystem

import itertools
import torch
import torch.multiprocessing as mp
import csv

from morl.agents import MOPPOAgent
from morl.networks import MOPPO_Actor_Critic, FDDTN
import copy

def global_reset_eval(args, evaluation, eval_inj_rate=0.5, wl_file=None):
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

    chiplet_env_for_RL = chiplets_RL_env.Chiplets_rl_env(simpy_env=simpy_env, 
                                                         args=args)
    
    if evaluation:
        simpy_env.process(chiplet_sim.execute_simulation_rl_disc_clustered())
    return simpy_env, chiplet_sim, chiplet_env_for_RL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate_critic", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--learning_rate_actor", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--eval_injection_rate", type=float, default=[0.3],  help="injection rate")
    parser.add_argument("--hidden_size", type=int, default = 64, help="number of neurons in a layer")
    parser.add_argument("--seed", type=int, default = 1, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--load_model", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, nargs="?", const=True, help="load previously trained model")
    parser.add_argument("--model_path", type=str, default=None, help="path to the model")

    # Chiplet system specific arguments
    parser.add_argument("--n_chiplets", type=int, default=78, help="number of chiplets in the system")
    parser.add_argument("--n_clusters", type=int, default=4, help="number of clusters in the system")
    parser.add_argument("--adj_matrix_file", type=str, default='params/adj_matrix_86x86.csv', help="adjacency matrix file")
    parser.add_argument("--chiplet_mapping_file", type=str, default='params/chiplet_mapping_78.yaml', help="chiplet mapping file")

    # DT specific arguments 
    parser.add_argument("--tree_depth", type=int, default=6, help="depth of the decision tree")
    parser.add_argument("--tree_alpha", type=int, default=5, help="alpha value used in DT")

    # Algorithm specific arguments
    parser.add_argument("--num_steps", type=int, default = 1024, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num_minibatches", type=int, default = 16, help="the number of mini-batches")

    args = parser.parse_args()
    
    args.batch_size = int( args.num_steps) 
    args.minibatch_size = int(args.batch_size // args.num_minibatches) 

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
        next_obs    = torch.Tensor(next_obs).to(args.device)

        with torch.no_grad():
            action, _ = agent.get_action_and_value(next_obs.unsqueeze(0), preference=preference_eval.unsqueeze(0), deterministic = True, mask=common.rl_state_info.supported_clusters)

        common.rl_state_info.action = action

        yield simpy_env.process(chiplet_env_for_RL.step(common.rl_state_info.action))
        reward      = common.rl_state_info.reward
        done        = common.rl_state_info.done

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

    device = torch.device("cpu")
    args.device = device

    #Initialize environment related arguments
    args.obs_shape = chiplet_env_for_RL.observation_space.shape[0]
    args.action_shape = chiplet_env_for_RL.action_space.n
    args.reward_size = 2

    ### Initialize Networks
    model = MOPPO_Actor_Critic(args, FDDTN(args))

    actor = model.actor
    critic = model.critic
    agent = MOPPOAgent(actor, critic, args).to(args.device)

    args.w_step_size = 0.5
    w_batch_test = generate_w_batch_test(args,step_size = args.w_step_size)
    print(f"Preference Space: {w_batch_test}")

    agent.w_batch = w_batch_test
    
    if args.load_model:
        if args.model_path:
            agent_actor = torch.load(args.model_path)
            id = args.model_path.split('_')[-1].split('.')[0]
        else:
            agent_actor = torch.load(f'final_model/thermos_mesh_best_actor.pkl')
            id = '1'

        agent.actor.load_state_dict(agent_actor)

    if not os.path.exists(f'rl_results/test/{method}'):
        os.makedirs(f'rl_results/test/{method}')

    start_time = time.time()
    run_simulation = 1
    
    # if eval_inj_rate is a single value and not a list
    if not isinstance(args.eval_injection_rate, list):
        args.eval_injection_rate = [args.eval_injection_rate]

    for inj_rate in args.eval_injection_rate:
        pref_cnt = 0
        eval_objs = np.zeros((w_batch_test.shape))
        for preference_eval in w_batch_test:
            eval_cnt = 0
            job_exe_time_eval= []
            job_energy_consumption_eval= []

            workloads = [f'{network}_500_1000-20000.csv']

            for workload in workloads:
                for _ in range(run_simulation):
                    common.rl_seed = eval_cnt

                    workload_file = f'workloads/{workload}'
                    simpy_env, chiplet_sim, chiplet_env_for_RL = global_reset_eval(args, evaluation = True, eval_inj_rate=inj_rate, wl_file=workload_file)

                    ## For first task, preemptively trigger event so that simulation doesn't hang
                    common.rl_state_info.wait_for_ptask.succeed()

                    simpy_env.process(run_episode_eval(chiplet_env_for_RL, simpy_env, preference_eval, agent, args))

                    simpy_env.run(until = common.rl_state_info.sim_done)

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
                    with open(f'rl_results/test/{method}/test_{wl_name}_{inj_rate}_{method}_results.csv', mode='a') as file:
                        file_w = csv.writer(file)
                        if file.tell() == 0:
                            file_w.writerow(['ID', 'preference', 'completion_rate (jobs/s)', 
                                             'total_neural_nets', 'average_execution_time (s)', 
                                               'average_energy_consumed (J)', 'Return', 'Iterations'])
                        file_w.writerow([id, preference_eval, completion_rate_jobs, 
                                         common.results.completed_jobs, common.results.average_execution_time, 
                                         common.results.average_energy_consumption, common.results.episode_return, common.results.episode_iterations])
                    
            eval_job_exe_time = np.array(job_exe_time_eval).mean()
            eval_energy_consumption = np.array(job_energy_consumption_eval).mean()

            eval_objs[pref_cnt,:] = np.array([eval_job_exe_time,eval_energy_consumption])

            print(f'[I] Completed iteration with preference {preference_eval}: avg execution time:{eval_job_exe_time}, energy consumption:{eval_energy_consumption}')

            pref_cnt +=1

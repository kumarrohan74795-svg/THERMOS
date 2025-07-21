import numpy as np
import simpy
from gym import spaces
from morl import common

## class Chiplets_env(gym.Env):
class Chiplets_rl_env() :

    def __init__(self, simpy_env, args) :

        ####################################################
        ## chiplet system Initialization
        ####################################################
        ## Initialize Simpy environment
        self.simpy_env = simpy_env

        self.n_chiplet = args.n_chiplets
        self.n_clusters = args.n_clusters

        self.observations = self.n_clusters * 3 + 4 + 4

        self.observation_space = spaces.Box(low=np.zeros((self.observations,)), high=20*np.ones((self.observations,)), dtype=np.float32)

        self.action_space = spaces.Discrete(self.n_clusters)

        ## Initialization
        self.current_state = np.zeros([self.observations])
        self.terminal = False
        self.step_idx = 0
        
    def wait_for_sched(self) :
        yield common.rl_state_info.wait_for_sched

    def reset(self):
        '''
            reset the environment to initial state
        '''
        self.current_state = np.zeros([self.observations])
        self.terminal = False
        self.step_idx = 0
        
        return self.current_state

    def step(self, action):
        '''
            step one move and feed back reward
        '''
        if common.DEBUG_RL :
            print('[ INFO] [STEP ] Waiting for scheduler to complete...')

        ## Signal to scheduler that action is ready
        common.rl_state_info.wait_for_action.succeed()

        ## Wait for reward to be generated from scheduler
        yield common.rl_state_info.wait_for_sched
        common.rl_state_info.wait_for_sched = self.simpy_env.event()

        return 
    
    def observe(self):
        ''' reset the enviroment '''
        return self.current_state


if __name__ == "__main__" :

    ## Create a simpy environment
    env = simpy.Environment()

    ## Create an instance of the Chiplets environment
    chiplets_env = Chiplets_rl_env(env)

    print(chiplets_env.observation_space.shape)


    common.rl_state_info = common.RLStateInfo()

    ## Add synchronization events
    common.rl_state_info.wait_for_task   = env.event()
    common.rl_state_info.wait_for_action = env.event()
    common.rl_state_info.wait_for_sched  = env.event()
    common.rl_state_info.wait_for_ptask  = env.event()
    common.rl_state_info.sim_done        = env.event()


    chiplets_env.step(10)

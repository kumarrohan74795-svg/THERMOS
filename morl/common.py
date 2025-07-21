import numpy as np
import yaml

DEBUG_RL = False

# =============================================================================
## Class to store matrics and state for RL
class RLStateInfo:
    def __init__(self, chiplets=22, observations=70) : #70
        self.observations    = observations
        self.current_state   = np.zeros(self.observations) 
        self.imm_next_state  = np.zeros(self.observations) # not used
        self.action          = np.zeros(chiplets)
        self.reward          = 0
        self.supported_clusters = []
        self.terminal        = 0
        self.done            = 0
        self.now             = 0
        
        ## Handshake events
        self.wait_for_action = 0 # wait for action
        self.wait_for_task   = 0 # wait for task
        self.wait_for_sched  = 0 # wait for scheduler
        self.wait_for_ptask  = 0 # wait for previous task
        self.sim_done        = 0

class PerfStatics:
    '''!
    Define the PerfStatics class to calculate energy consumption and total execution time.
    '''
    def __init__(self):
        self.simulation_time = 0.0                   # The total simulation time (s)
        self.variance_energy_consumption = 0.0       # Variance of the energy consumption of completed tasks (us)
        self.total_injected_jobs = 0                 # Count the total number of jobs that are injected into the buffer, this might be more than the total number of jobs because of waiting jobs
        self.completed_jobs = 0                      # Count the number of jobs that are completed/retired 
        self.episode_return = 0                      # The total reward of the episode
        self.episode_iterations = 0                  # The total number of iterations in the episode
        self.execution_time_list = []                # List of execution time for each job
        self.energy_consumption_list = []            # List of energy consumption for each job

    def calculate_var_and_array(self):
        '''!
        Calculate the variance of the execution time and energy consumption of completed tasks.
        '''
        self.execution_time_list = np.array(self.execution_time_list)
        self.average_execution_time = np.average(self.execution_time_list)
        
        self.energy_consumption_list = np.array(self.energy_consumption_list)
        self.average_energy_consumption = np.average(self.energy_consumption_list)

global results

# load material properties from yaml file
def load_dict_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        dict = yaml.safe_load(f)
    return dict

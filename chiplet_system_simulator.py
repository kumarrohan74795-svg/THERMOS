
import os
import math
import numpy as np
import networkx as nx
from datetime import datetime

import random

from chiplet_class import Chiplet
from mapping_operation_class import MappingOperation
from entire_network_class import EntireNetwork

from params.network_definitions import networks

from morl.common import load_dict_yaml
from morl import common

# Import functions from simulation_functions.py
from simulation_functions import (
    calculate_layer_requirements,
    process_network,
    build_chiplet_network,
)

from mapping_functions import nn_huristic_for_2nd_level_mapping

# Import DSS class and chiplet_node_placement
from thermal_model.dss_class import DSS, chiplet_node_placement, chiplet_clusters, io_nodes

class ChipletSystem:
    """
    Models and manages a cluster of chiplets, tracking crossbar availability, simulation time, and temperature.
    """
    def __init__(
        self,
        simpy_env,
        chiplet_mapping_file,
        adj_matrix_file,
        network_list=['ResNet18', 'ResNet50','EfficientNet-B3', 'AlexNet', 'Inception-v3', 'MobileNetV3-Large'],
        network_probabilities=None,
        inj_rate=0.5,
        link_params=None,
        mapper='rl', # simba, rl, big_little
        time_step=0.1, # in seconds
        test=False,
        total_jobs=100, # only for training
        wl_file=None,
        warmup_jobs=25, # number of warmup jobs
        job_buffer_size=20, # number of jobs to buffer
    ):
        """
        Initializes the ChipletSystem with chiplet configurations and simulation parameters.

        Parameters:
            simpy_env (simpy.Environment): SimPy environment for simulation.
            chiplet_mapping_file (str): YAML file containing chiplet mapping.
            adj_matrix_file (str): File containing the adjacency matrix.
            network_list (list): List of available network names.
            link_params (dict): Link parameters.
            total_jobs (int): Number of networks to simulate.
            time_step (float): Time step for each iteration in seconds.
            inj_rate (float): Probability of not generating a job at a time step.
            network_probabilities (list): List of probabilities corresponding to each network.
            mapper (str): Mapper to use for network mapping.['rl', 'big_little', etc]
            test (bool): True if testing 
            wl_file (str): Path to the workload file. Only used if test is True.
            warmup_jobs (int): Number of warmup jobs to run before collecting stats.
            job_buffer_size (int): Maximum number of jobs to buffer.
        """
        # Initialize SimPy environment
        self.simpy_env = simpy_env  # For future use with SimPy
        
        # Load chiplet mapping from YAML
        self.chiplet_mapping = load_dict_yaml(chiplet_mapping_file)
        self.num_chiplets = len(self.chiplet_mapping)
        self.chiplets = [
            Chiplet(chiplet_id=i + 1, chiplet_type=self.chiplet_mapping[i + 1])
            for i in range(self.num_chiplets)
        ]

        self.io_chiplets = io_nodes.keys()

        # Build the chiplet network
        self.chiplet_network = build_chiplet_network(adj_matrix_file, self.chiplet_mapping, self.io_chiplets)

        total_chiplets = self.num_chiplets + len(self.io_chiplets)

        shortest_distance = np.full((total_chiplets, total_chiplets), np.inf)

        for source in range(total_chiplets):
            lengths = nx.shortest_path_length(self.chiplet_network, source=source+1)
            for target, length in lengths.items():
                shortest_distance[source][target-1] = length
        
        self.distance_matrix = shortest_distance

        # Normalize the shortest distance
        self.shortest_distance = (max(shortest_distance.flatten()) - shortest_distance)/max(shortest_distance.flatten())

        self.network_list = network_list
        self.network_probabilities = network_probabilities
        self.inj_rate = inj_rate

        # link parameters for communication from UCIe
        self.link_params = link_params or {
            'link_width': 64,
            'link_frequency': 1e9,
            'energy_per_bit': 0.5*1e-12,
            'per_hop_latency': 2*1e-9,
        }

        self.mapper = mapper
        # check if the mapper is valid
        if self.mapper not in ['rl']:
            if common.DEBUG_RL:
                print(f"Error: Invalid mapper '{self.mapper}'.")
            exit(1)

        self.time_step = time_step
        self.test = test # Is it test or training, depending on this the job generator will be different

        if self.test:
            # check if workload file exists
            if not os.path.exists(wl_file):
                if common.DEBUG_RL:
                    print(f"Error: Workload file '{wl_file}' does not exist.")
                exit(1)
            # read the workload file in list, exclude the 1st row
            with open(wl_file, 'r') as f:
                self.workload = f.readlines()[1:]
            
            self.total_jobs_to_run = len(self.workload)
        else:
            # Total number of networks to simulate (only for training)
            self.total_jobs_to_run = total_jobs

        self.warmup_jobs = warmup_jobs

        # Number of jobs to buffer
        self.job_buffer_size = job_buffer_size

        # Initialize DSS model for temperature tracking
        self.disc_A = np.loadtxt('thermal_model/disc_A_matrix.csv', delimiter=',')
        self.disc_B = np.loadtxt('thermal_model/disc_B_matrix.csv', delimiter=',')
        self.temperature_initial = np.zeros(self.disc_A.shape[0])

        # Use the imported chiplet_node_placement
        self.chiplet_nodes = chiplet_node_placement

        # cluster number to chiplet mapping, index starts from 0
        self.chiplet_clusters = chiplet_clusters

        # Initialize DSS model
        self.dss_model = DSS(disc_A=self.disc_A, disc_B=self.disc_B, chiplet_nodes=self.chiplet_nodes, ts=self.time_step, tp=self.time_step)

        # Initialize the current temperature state
        temperature_now, chiplet_temperature = self.dss_model.run_dss(self.temperature_initial, np.zeros(self.num_chiplets))
        self.temperature_initial = temperature_now
        self.current_temperature_state = chiplet_temperature.copy()

        # Initialize the threshold temperature for each chiplet
        self.threshold_temp = np.zeros(self.num_chiplets)
        for chiplet in self.chiplets:
            self.threshold_temp[chiplet.id - 1] = chiplet.threshold_temp 

        self.global_clock = 0.0  # Initialize global clock to zero

        self.warmup_period = 0.0  # Initialize warmup period to zero

        self.next_mapping_is_first_layer = True

        # List to keep track of active mapping operations
        self.active_mappings = []        

        self.reward_mapping_queue = []

        self.prev_chiplet_power = self.compute_total_chiplet_power()

        self.all_jobs_done = False

        # counters to keep track of warmup jobs
        self.injected_job_counter = 0
        self.total_admitted_jobs = 0
        self.completed_job_counter = 0

        # Initialize completed mappings
        self.completed_mappings = []

        # Only contains mappable network in the current cycle
        self.job_current_cycle = []

        # Network queue to store the network that can't be mapped in the current cycle
        self.job_queue = []
        
        # Get the total amount of memory in the system, in bits
        self.system_memory = 0
        
        for chiplet in self.chiplets:
            self.system_memory += chiplet.get_total_memory()

        self.system_memory_clusters = np.array([np.sum(self.get_available_memory_in_system()[cluster-1]) for cluster in self.chiplet_clusters.values()])
        self.system_crossbars_clusters = np.array([np.sum(self.get_available_crossbars_per_chiplet()[cluster-1]) for cluster in self.chiplet_clusters.values()])

        self.system_crossbars_chiplets = self.get_available_crossbars_per_chiplet()

        self.get_memory_per_crossbar()

        # Intitialize statistics storage          
        common.results = common.PerfStatics()

    def get_available_memory_in_system(self):
        """
        Retrieves the current number of available memory in int8 units in each chiplet.

        Returns:
            int: Total available memory in the system.
        """
        available_memory = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            available_memory[chiplet_id - 1] = chiplet.get_available_memory()
        
        return available_memory

    def get_available_crossbars_per_chiplet(self):
        """
        Retrieves the current number of available crossbars in each chiplet.

        Returns:
            numpy.ndarray: Array where each element corresponds to the number of available crossbars in a chiplet.
        """
        crossbar_availability = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            available_crossbars = chiplet.get_available_crossbars()
            crossbar_availability[chiplet_id - 1] = available_crossbars

        return crossbar_availability
    
    def get_memory_per_crossbar(self):
        """
        Retrieves the memory per crossbar in the system.

        Returns:
            numpy.ndarray: Array where each element corresponds to the memory per crossbar in a chiplet.
        """
        self.memory_per_crossbar = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            self.memory_per_crossbar[chiplet_id - 1] = chiplet.memory_per_crossbar        

    def compute_total_chiplet_power(self):
        """
        Calculates the current power state of each chiplet based on the active or idle state of each crossbar.

        Returns:
            numpy.ndarray: Array of total power consumption for each chiplet 
        """
        chiplet_power = np.zeros(self.num_chiplets)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            power = chiplet.calculate_current_power()
            chiplet_power[chiplet_id-1] = power
        return chiplet_power

    def advance_simulation_time(self, time_elapsed_s):
        """
        Simulates the passing of time by decrementing each busy time in the chiplets
        and updating active mappings.

        Args:
            time_elapsed_s (float): The amount of time to decrement (in seconds).
        """
        # Increment global clock by time_step
        self.global_clock += time_elapsed_s

        # Update crossbar busy times and availability
        for chiplet in self.chiplets:
            chiplet.crossbar_busy_times = chiplet.crossbar_busy_times - time_elapsed_s
            chiplet.crossbar_busy_times = np.maximum(chiplet.crossbar_busy_times, 0.0)

            chiplet.crossbar_available[chiplet.crossbar_busy_times == 0] = 1  # Available

        # Update remaining time for active mappings
        for mapping in self.active_mappings:
            if not mapping.completed:
                mapping.update_remaining_time(time_elapsed_s)

    def refresh_system_temperature(self):
        """
        Updates the temperature of the system using the DSS model.
        """
        power_sequence = self.prev_chiplet_power

        # Run the DSS model
        temperature_now, chiplet_temperature = self.dss_model.run_dss(self.temperature_initial, power_sequence)

        # Update the initial temperature for the next iteration
        self.temperature_initial = temperature_now

        # Overwrite the current temperature state
        self.current_temperature_state = chiplet_temperature.copy()

        # Get current chiplet temperatures
        chiplet_temperatures = self.current_temperature_state

        temperature_violation = chiplet_temperatures > self.threshold_temp

        # check if any chiplet temperature is above threshold
        if not any(temperature_violation):
            for entire_mapping in self.active_mappings:
                for operation in entire_mapping.mapping_operations:
                    if operation.stalled:
                        operation.resume_this_mapping()

        # check all the chiplets whose temperature is above threshold, check all active mappings and if any of the chiplet is in the mapping, stall the mapping
        for chiplet_id in np.where(temperature_violation)[0]:
            for entire_mapping in self.active_mappings:
                for operation in entire_mapping.mapping_operations:
                    if (not operation.completed) and (not operation.stalled) and (operation.network_crossbar_required[chiplet_id] > 0):
                        operation.stall_this_mapping(self.time_step)
    
    def can_map_layer(self, network_name, network_layer_info, available_crossbars):
        """
        Check the total memory/crossbars required for the network and if they are available in the chiplets.
        """

        available_memory_atm = self.memory_per_crossbar * available_crossbars

        required_memory = network_layer_info['num_weights']
        # Because not all crossbars are 100% utilized 
        if ('conv3_' in network_layer_info['name']) or ('head_conv' in network_layer_info['name']) or ('conv_bottleneck' in network_layer_info['name']):
            multiplier = 0.1
        elif 'conv' in network_layer_info['name']:
            multiplier = 0.3
        else:
            multiplier = 0.7
        
        if required_memory > multiplier*available_memory_atm.sum():
            if common.DEBUG_RL:
                print(f'required memory: {required_memory} available memory: {available_memory_atm.sum()}')
            return False
        else:
            return True
        
    def transfer_job_queue_to_current_cycle(self):
        """
        Transfer the first job in the job_queue to the current cycle.
        """
        self.job_current_cycle = self.job_queue.pop(0)

    def process_and_transfter_network(self, network_name, total_inputs):
        """
        Process the network and transfer it to the current cycle if it can be mapped.
        Input:
            network_name: Name of the network to map.
            total_inputs: Total inputs to the network (between 10-100).
        """
        network_config = networks.get(network_name, None)
        if network_config is None:
            if common.DEBUG_RL:
                print(f"Error: Network '{network_name}' is not defined")
            exit(1)

        network_layers_info = process_network(network_config)

        current_time = self.global_clock

        job_list = [network_name, network_layers_info, total_inputs, current_time]

        self.job_queue.append(job_list)
    
    def generate_test_job(self):
        """
        Generate a job from the workload file
        """
        # get the network from the workload file
        if random.random() < self.inj_rate:
            if self.workload:
                # get the 1st element
                job = self.workload.pop(0)
                job = job.split(',')
                network_name = job[0]
                total_inputs = int(job[1])

                self.injected_job_counter += 1

                self.process_and_transfter_network(network_name, total_inputs)
            else:
                # no more jobs in the workload file
                self.all_jobs_done = True

    def generate_random_job(self):
        """
        randomly generates a job for RL training
        """
        if self.injected_job_counter == self.total_jobs_to_run:
            self.all_jobs_done = True
        else:
            if random.random() < self.inj_rate:
                no_inputs = random.randint(1000, 20000)
                if self.network_probabilities is None:
                    # Uniform probability if not provided
                    network_name = random.choice(self.network_list)
                else:
                    network_name = random.choices(
                        self.network_list, weights=self.network_probabilities, k=1
                    )[0]

                self.injected_job_counter += 1
                
                # if the network is mappable, transfer the job to the current cycle
                self.process_and_transfter_network(network_name, no_inputs)

    def generate_network_job(self):
        """
        Simulates a job generator that, at each time step, either provides a network to map or not.

        Returns:
            tuple: (network_name (str or None), total_inputs (int or None))
                - network_name: The name of the network to map, or None if no job is generated.
                - total_inputs: Total inputs to the network (currently fixed at 1).
        """
        # seed random number generator
        random.seed(datetime.now().microsecond)

        if len(self.job_queue) < self.job_buffer_size:
            if self.test:
                self.generate_test_job()
            else: # training
                self.generate_random_job()

        # if self.job_current_cycle is empty, generate a new mapping list
        if not self.job_current_cycle:
            if self.job_queue:
                self.transfer_job_queue_to_current_cycle()

    def check_for_completed_mappings_and_update_performance(self):

        # Check for completed mappings
        self.completed_now = []
        for entire_mapping in self.active_mappings:
            # reduce power for completed mappings in entire network
            entire_mapping.reduce_power_chiplets()
            
            if entire_mapping.completed:
                self.completed_mappings.append(entire_mapping)
                self.completed_now.append(entire_mapping)

        # Remove completed mappings from active_mappings
        for entire_mapping in self.completed_now:
            self.active_mappings.remove(entire_mapping)
            
            self.completed_job_counter += 1

            if self.completed_job_counter == self.warmup_jobs:
                self.warmup_period = self.global_clock
                
            if self.completed_job_counter > self.warmup_jobs:
                # Update performance stats here
                common.results.execution_time_list.append(entire_mapping.total_latency)
                common.results.energy_consumption_list.append(entire_mapping.total_energy)

                # Increment completed jobs count
                common.results.completed_jobs += 1

    def execute_mapping_simulation(self, network_name, layer_mappings, total_inputs, waiting_time, last_layer):
        """
        Initiates the mapping of a network by simulating its mapping to chiplets.
        Returns the latency and energy immediately, and tracks the mapping operation
        for temperature impact, which is determined after the mapping completes.

        Parameters:
            network_name (str): Name of the network to map.
            layer_mappings 
            total_inputs (int): Total inputs to the network.
            waiting_time (float): Time the network has been waiting in the queue.

        Returns:
            MappingOperation: An object representing the mapping operation.
                Contains latency, energy, and will provide temperature upon completion.
        """
        # Create a MappingOperation call functions to simulate mapping
        mapping_operation = MappingOperation(
            network_name=network_name,
            chiplets=self.chiplets,
            network=self.chiplet_network,
            io_chiplets=self.io_chiplets,
            total_inputs=total_inputs,
            layer_mappings=layer_mappings,
            link_params=self.link_params,
            waiting_time=waiting_time
        )

        mapping_operation.map_network_and_compute_latency_energy()

        mapping_operation.update_busy_times_and_power()

        if self.next_mapping_is_first_layer:
            entire_mapping = EntireNetwork(network_name, total_inputs)

            # Add the mapping operation to the cluster's active mappings
            self.active_mappings.append(entire_mapping)
            
            self.reward_mapping_queue.append(entire_mapping)
            
            self.total_admitted_jobs += 1

        self.active_mappings[-1].add_operation(mapping_operation, last_layer)

        if last_layer:
            # Clear job_current_cycle after mapping last layer
            self.job_current_cycle = []
            self.next_mapping_is_first_layer = True
        else:
            self.next_mapping_is_first_layer = False

        # Return the mapping operation (contains latency and energy)
        return mapping_operation

    def execute_simulation_rl_disc_clustered(self):
        """
        Runs the simulation for the specified number of iterations. only for RL

        Returns:
            List of completed mapping operations with their latency, energy, and temperature at completion.
        """
        iteration = 0

        # Initialize performance statistics
        common.results.simulation_time = 0.0
        common.results.completed_jobs = 0

        while True:
            iteration += 1
            # Update temperatures based on power from previous time step
            self.refresh_system_temperature()

            # Generate a job using the job generator
            self.generate_network_job()
            if self.job_current_cycle:
                network_name, network_layers_info, total_inputs, enqueue_time = self.job_current_cycle

                if common.DEBUG_RL:
                    for layer in network_layers_info:
                        print(f"Layer: {layer['name']}, Weights: {layer['num_weights']}, MACs: {layer['total_macs']}, Comm: {layer['input_activation']}")
            else:
                network_name = None

            if network_name is not None:

                if common.DEBUG_RL:
                    print(f"Network '{network_name}', num_inputs: {total_inputs} injected this iteration.")

                mapping_failed = False
                last_layer = False
                layer_mappings = []  # Collect layer mappings for the entire network

                # Initialize tracker for crossbar usage
                available_crossbars = self.get_available_crossbars_per_chiplet()

                if self.mapper != 'rl':
                    if common.DEBUG_RL:
                        print(f"Error: Invalid mapper '{self.mapper}' for RL simulation.")
                    exit(1)
                else:
                    if self.next_mapping_is_first_layer:
                        common.results.total_injected_jobs += 1

                    total_layer = len(network_layers_info)
                    
                    current_temperatures_chiplet = (self.current_temperature_state - 300)/55.0 # Normalized temperatures
                    current_temperatures_cluster = [np.max(current_temperatures_chiplet[indices-1]) for indices in self.chiplet_clusters.values()]

                    for network_layer_idx in range(len(network_layers_info)):
                        if not self.can_map_layer(network_name, network_layers_info[network_layer_idx], available_crossbars):
                            # remove all the previous layers 
                            self.job_current_cycle = [network_name, network_layers_info[network_layer_idx:], total_inputs, enqueue_time]
                            break
                        else:
                            if network_layer_idx == len(network_layers_info) - 1:
                                last_layer = True

                        percentage_remaining = 100
                        mapping_scheme = []
                        
                        # previous mapping
                        previous_mapping = np.zeros(self.num_chiplets)

                        if network_layer_idx > 0:
                            for mapping in layer_mappings[network_layer_idx-1][1]:
                                chiplet_id, percentage = mapping
                                previous_mapping[chiplet_id-1] = percentage/100.0
                        
                        previous_mapping_cluster = [np.sum(previous_mapping[indices-1]) for indices in self.chiplet_clusters.values()]

                        while (percentage_remaining > 0) and (not mapping_failed):
                            layer_features = []
                            # Features for the layer
                            layer_features.append(network_layers_info[network_layer_idx]['num_weights']*percentage_remaining/100000000.0)
                            layer_features.append(network_layers_info[network_layer_idx]['total_macs']*percentage_remaining/10000000000.0)
                            layer_features.append(network_layers_info[network_layer_idx]['input_activation']/100000.0)
                            layer_features.append(total_inputs/20000.0) # Total inputs to the network

                            network_features = []
                            network_features.append((total_layer-network_layer_idx)/total_layer) # remaining layers
                            # remaining weights after this layer
                            network_features.append(np.sum([layer['num_weights'] for layer in network_layers_info[network_layer_idx:]])/100000000.0)
                            # remaining MACs after this layer
                            network_features.append(np.sum([layer['total_macs'] for layer in network_layers_info[network_layer_idx:]])/1000000000.0)
                            # remaining communication after this layer
                            network_features.append(np.sum([layer['input_activation'] for layer in network_layers_info[network_layer_idx:]])/100000.0)

                            # This is available memory
                            available_memory = available_crossbars
                            available_memory_cluster = np.array([np.sum(available_memory[indices-1]) for indices in self.chiplet_clusters.values()])/self.system_crossbars_clusters

                            common.rl_state_info.current_state = list(previous_mapping_cluster)
                            common.rl_state_info.current_state.extend(list(available_memory_cluster))
                            common.rl_state_info.current_state.extend(list(current_temperatures_cluster))
                            common.rl_state_info.current_state.extend(layer_features)
                            common.rl_state_info.current_state.extend(network_features)

                            # supported clusters are where available crossbars are greater than 0
                            common.rl_state_info.supported_clusters = [cluster for cluster in range(len(self.chiplet_clusters)) if available_memory_cluster[cluster] > 0]

                            # Handshake with the RL agent; wait for previous task to complete
                            yield common.rl_state_info.wait_for_ptask
                            common.rl_state_info.wait_for_ptask = self.simpy_env.event()

                            # Handshake with the RL agent, task is available for scheduling decision
                            common.rl_state_info.wait_for_task.succeed()

                            # Handshake with the RL agent, action is available for scheduling decision
                            yield common.rl_state_info.wait_for_action
                            common.rl_state_info.wait_for_action = self.simpy_env.event()

                            action = common.rl_state_info.action.item()

                            available_crossbars_cluster = np.sum(available_crossbars[self.chiplet_clusters[action]-1])

                            if available_crossbars_cluster <= 0:
                                common.rl_state_info.reward = np.array([0, 0])
                                common.rl_state_info.terminal = 1
                                common.rl_state_info.wait_for_sched.succeed()
                                mapping_failed = True
                                break

                            crossbars_needed_total = calculate_layer_requirements(network_layers_info[network_layer_idx], self.chiplets[self.chiplet_clusters[action][0]-1])

                            crossbars_needed_this_time = math.ceil(crossbars_needed_total*percentage_remaining/100)

                            if crossbars_needed_this_time > available_crossbars_cluster: # not enough crossbars
                                can_alloc_cb = available_crossbars_cluster
                                percentage_alloc = (available_crossbars_cluster/crossbars_needed_total)*100
                            else:
                                can_alloc_cb = crossbars_needed_this_time
                                percentage_alloc = percentage_remaining

                            percentage_remaining = percentage_remaining - percentage_alloc

                            available_crossbars, mapping_this_action = nn_huristic_for_2nd_level_mapping(available_crossbars=available_crossbars,
                                                                                                         allocate_crossbars=can_alloc_cb,
                                                                                                         allocate_percentage=percentage_alloc,
                                                                                                         chiplets_in_this_cluster=self.chiplet_clusters[action]-1,
                                                                                                         distance_matrix=self.distance_matrix,
                                                                                                         previous_mapping=previous_mapping)
                            
                            mapping_scheme.extend(mapping_this_action)
                            
                            if percentage_remaining > 0 and np.sum(available_crossbars) == 0:
                                common.rl_state_info.reward = np.array([0, 0])
                                common.rl_state_info.terminal = 1
                                common.rl_state_info.wait_for_sched.succeed()
                                mapping_failed = True
                                break

                            if (last_layer) and (percentage_remaining == 0):
                                # for the last layer, we need to give actual reward
                                pass 
                            else:
                                energy_latency = np.array([0.0, 0.0])
                                for mapping in self.reward_mapping_queue:
                                    if mapping.completed:
                                        energy_latency = np.array([-0.22*mapping.extra_energy, -mapping.extra_latency])
                                        self.reward_mapping_queue.remove(mapping)
                                        break

                                common.rl_state_info.reward = energy_latency
                                common.rl_state_info.terminal = 0
                                common.rl_state_info.wait_for_sched.succeed()

                        layer_mappings.append([network_layers_info[network_layer_idx], mapping_scheme])

                if mapping_failed:
                    if common.DEBUG_RL:
                        print(f"Mapping failed for network '{network_name}', reverting system state and skipping network.")

                elif layer_mappings:
                    # waiting time is the time the network waited in the queue
                    waiting_time = self.global_clock - enqueue_time

                    # Simulate mapping for the network
                    mapping_operation = self.execute_mapping_simulation(network_name, layer_mappings, total_inputs, waiting_time, last_layer)

                    # **Print Mapping Information**
                    if common.DEBUG_RL:
                        print(f"\nMapped Network '{network_name}' to Chiplets:")
                        for lm in layer_mappings:
                            print(f"Layer '{lm[0]['name']}':")
                            for mapping in lm[1]:
                                chiplet_num = mapping[0]
                                percentage = mapping[1]
                                print(f"  - Chiplet {chiplet_num}: {percentage}% of layer")
                        print(f"Total Mapping Latency: {mapping_operation.total_latency:.6e}s")
                        print(f"Total Mapping Energy: {mapping_operation.total_energy:.6e}J")
                    
                    if last_layer:
                        # Assign reward for the entire network mapping
                        # 0.22 is for energy scaling factor, such that reward is in the same scale as latency
                        energy = 0.22*self.active_mappings[-1].total_energy 
                        latency = self.active_mappings[-1].total_latency
                        reward = np.array([-energy, -latency])

                        common.rl_state_info.reward = reward
                        common.rl_state_info.terminal = 1
                        common.rl_state_info.wait_for_sched.succeed()

            else:
                # No network mapping this iteration
                if common.DEBUG_RL:
                    print(f"No network mapped this iteration {iteration}.")

            self.prev_chiplet_power = self.compute_total_chiplet_power()

            self.check_for_completed_mappings_and_update_performance()

            # Decrement crossbar busy times by time_step
            self.advance_simulation_time(self.time_step)

            if self.test:
                end = ((common.results.total_injected_jobs - 50) == self.total_jobs_to_run) or \
                (self.all_jobs_done and (self.completed_job_counter == self.injected_job_counter))
            else:
                end = self.all_jobs_done and (self.completed_job_counter == self.injected_job_counter)

            if end:

                common.results.simulation_time = self.global_clock - self.warmup_period
                common.results.calculate_var_and_array()

                if common.DEBUG_RL:

                    print("\n=== Simulation Completed ===")
                    for idx, mapping in enumerate(self.completed_mappings, start=1):
                        print(f"\nMapping {idx}:")
                        print(f"Network Name: {mapping.network_name}")
                        print(f"Total Latency: {mapping.total_latency:.6e}s")
                        print(f"Total Energy: {mapping.total_energy:.6e}J")
                # Signal simulation completion
                common.rl_state_info.done = 1
                common.rl_state_info.sim_done.succeed()
                break

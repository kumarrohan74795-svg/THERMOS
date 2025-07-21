import numpy as np
from morl import common
import math
from simulation_functions import calculate_layer_requirements
import networkx as nx

class MappingOperation:
    """
    Represents a mapping operation of a network.
    """

    def __init__(self, network_name, chiplets, io_chiplets, network, total_inputs, layer_mappings, link_params, waiting_time):
        """
        Initializes the MappingOperation.

        Parameters:
            network_name (str): Name of the network being mapped.
            chiplets (list): List of chiplet objects in the system.
            network (nx.Graph): NetworkX graph representing the chiplet network.
            total_inputs (int): Total inputs to the network.
            layer_mappings :
                - Layer information
                - layer_mappings: List of layer mappings specifying chiplets and percentages.
            link_params (dict): Link parameters.
        """
        self.network_name = network_name # Name of the network
        self.chiplets = chiplets # List of chiplets
        self.io_chiplets = io_chiplets
        self.network = network # Chiplet network
        self.total_inputs = total_inputs # Total inputs (images) to the network
        self.layer_mappings = layer_mappings # Layer mappings
        self.link_params = link_params 

        self.active_xbars = []

        for chiplet in self.chiplets:
            self.active_xbars.append(np.zeros_like(chiplet.crossbar_available))

        self.total_latency = 0
        self.original_total_latency = 0 # Original total latency before stalling
        self.total_energy = 0
        self.time_remaining = 10**6  # Initialize with a large value
        self.networks_chiplets_energy = np.zeros(len(self.chiplets))
        self.network_active_energy = np.zeros(len(self.chiplets))
        self.network_crossbar_required = np.zeros(len(self.chiplets))

        self.leakae_power_per_crossbar = np.array([chiplet.idle_power_per_crossbar for chiplet in self.chiplets]) # Leakage energy per crossbar

        self.completed = False  # Flag to indicate completion
        self.activation_bits = 8  # Number of bits per activation
        self.waiting_time = waiting_time

        self.power_reduced = False
        self.stalled = False

    def map_network_and_compute_latency_energy(self):
        """
        Maps the network layers to chiplets and updates crossbar availability and busy times.
        """
        ### compute
        # calculate the compute time for each layer
        network_compute_times = np.zeros(len(self.layer_mappings)+1)
 
        # crossbar required for each chiplet
        network_crossbar_required = np.zeros(len(self.chiplets))
        network_compute_energy = np.zeros(len(self.chiplets))
        
        layer_index = 0
        for layer_info, mapping_scheme in self.layer_mappings:
            total_mac_ops = layer_info['total_macs']
            layer_compute_times = np.zeros(len(self.chiplets))
            for mapping in mapping_scheme:
                chiplet_id, percentage = mapping
                chiplet = next((c for c in self.chiplets if c.id == chiplet_id), None)
                if chiplet is None:
                    if common.DEBUG_RL:
                        print(f"Error: Chiplet ID {chiplet_id} not found during mapping.")
                    exit(1) # chiplet cannot be None
                
                # Calculate layer requirements for this chiplet
                total_crossbars_required_on_this_chiplets = calculate_layer_requirements(layer_info, chiplet)
                crossbar_used_this_chiplet = math.ceil(total_crossbars_required_on_this_chiplets * (percentage / 100))
                total_crossbars_this_chiplet = chiplet.crossbars_per_tile * chiplet.tiles_per_chiplet

                if crossbar_used_this_chiplet > total_crossbars_this_chiplet:
                    if (crossbar_used_this_chiplet-1) == total_crossbars_this_chiplet:
                        crossbar_used_this_chiplet -= 1
                    else:
                        if common.DEBUG_RL:
                            print(f"Error: Crossbars required for layer '{layer_info['name']}: {crossbar_used_this_chiplet}' exceeds total crossbars in chiplet {chiplet_id}. {total_crossbars_this_chiplet}")
                        exit(1)
                network_crossbar_required[chiplet_id - 1] = network_crossbar_required[chiplet_id - 1] + crossbar_used_this_chiplet

                # Calculate the time and energy for this chiplet
                mac_ops_this_chiplet = percentage * total_mac_ops/100
                compute_time_this_chiplet = mac_ops_this_chiplet / (chiplet.tops * 1e12)
                compute_energy_this_chiplet = mac_ops_this_chiplet * chiplet.energy_per_mac * 1e-15

                # update the layer_compute_times, layer_energy_consumptions, layer_crossbar_required
                layer_compute_times[chiplet_id - 1] = compute_time_this_chiplet
                network_compute_energy[chiplet_id - 1] = network_compute_energy[chiplet_id - 1] + compute_energy_this_chiplet
                
            # end for
            layer_compute_time = np.max(layer_compute_times)

            network_compute_times[layer_index] = layer_compute_time
            layer_index += 1
        
        network_compute_energy = network_compute_energy*self.total_inputs

        self.network_crossbar_required = network_crossbar_required

        ### communication

        # +1 because last layer needs to communicate with IO
        network_comm_times = np.zeros(len(self.layer_mappings)+1)
        weights_comm_times = np.zeros(len(self.layer_mappings)+1)

        # comm energy for each chiplet
        network_comm_energies = np.zeros(len(self.chiplets))

        # assumtion: communication -> compute -> communication
        for i in range(len(self.layer_mappings)+1):
            comm_act_latency_this_layer = 0
            comm_weights_latency_this_layer = 0

            if i == 0:
                # source is IO
                dst_mapping = self.layer_mappings[i][1]
                activation_pkts = self.layer_mappings[i][0]['input_activation'] * self.activation_bits / self.link_params['link_width']
                weights_pkts = self.layer_mappings[i][0]['num_weights'] * self.activation_bits / self.link_params['link_width']

                for next_mapping in dst_mapping:
                    comm_act_latency_this_chiplet = 0
                    comm_weights_latency_this_chiplet = 0

                    destination_chiplet_id, next_percentage = next_mapping
                    weights_pkts_this_chiplet = weights_pkts * next_percentage / 100
                    # send all activation packets to all chiplets, but weights only to the chiplet that needs it

                    min_distance = 10**6
                    for io_chiplet in self.io_chiplets:
                        comm_path = nx.shortest_path(self.network, source=io_chiplet, target=destination_chiplet_id)
                        if len(comm_path) < min_distance:
                            min_distance = len(comm_path)
                            source_chiplet_id = io_chiplet
                    
                    comm_path = nx.shortest_path(self.network, source=source_chiplet_id, target=destination_chiplet_id)
                    comm_path = comm_path[1:]

                    if len(comm_path) > 0:
                        comm_act_latency_this_chiplet = len(comm_path)* activation_pkts * self.link_params['per_hop_latency']
                        comm_weights_latency_this_chiplet = len(comm_path)* weights_pkts_this_chiplet * self.link_params['per_hop_latency']
                        
                        for chiplet_id in comm_path:
                            network_comm_energies[chiplet_id-1] += activation_pkts * self.link_params['link_width'] * self.link_params['energy_per_bit']
                            network_comm_energies[chiplet_id-1] += weights_pkts_this_chiplet * self.link_params['link_width'] * self.link_params['energy_per_bit']
                    
                    comm_act_latency_this_layer = max(comm_act_latency_this_layer, comm_act_latency_this_chiplet)
                    comm_weights_latency_this_layer = max(comm_weights_latency_this_layer, comm_weights_latency_this_chiplet)

            elif i == len(self.layer_mappings):
                # destination is IO
                src_mapping = self.layer_mappings[i-1][1]
                activation_pkts = self.layer_mappings[i-1][0]['output_activation'] * self.activation_bits / self.link_params['link_width']

                for mapping in src_mapping:
                    comm_act_latency_this_chiplet = 0

                    source_chiplet_id, percentage = mapping
                    activation_pkts_this_chiplet = activation_pkts * percentage / 100

                    min_distance = 10**6
                    for io_chiplet in self.io_chiplets:
                        comm_path = nx.shortest_path(self.network, source=source_chiplet_id, target=io_chiplet)
                        if len(comm_path) < min_distance:
                            min_distance = len(comm_path)
                            destination_chiplet_id = io_chiplet
                    
                    comm_path = nx.shortest_path(self.network, source=source_chiplet_id, target=destination_chiplet_id)
                    comm_path = comm_path[:-1]

                    if len(comm_path) > 0:
                        comm_act_latency_this_chiplet = len(comm_path)* activation_pkts_this_chiplet * self.link_params['per_hop_latency']

                        for chiplet_id in comm_path:
                            network_comm_energies[chiplet_id-1] += activation_pkts_this_chiplet * self.link_params['link_width'] * self.link_params['energy_per_bit']
                    
                    comm_act_latency_this_layer = max(comm_act_latency_this_layer, comm_act_latency_this_chiplet)

            else:
                src_mapping = self.layer_mappings[i-1][1]
                dst_mapping = self.layer_mappings[i][1]
                activation_pkts = self.layer_mappings[i][0]['input_activation'] * self.activation_bits / self.link_params['link_width']
                weights_pkts = self.layer_mappings[i][0]['num_weights'] * self.activation_bits / self.link_params['link_width']

                for mapping in src_mapping:
                    source_chiplet_id, percentage = mapping
                    activation_pkts_this_chiplet = activation_pkts * percentage / 100

                    # each chiplet of current layer will send data to each chiplet of next layer 
                    for next_mapping in dst_mapping:
                        comm_act_latency_this_chiplet = 0
                        destination_chiplet_id, next_percentage = next_mapping
                        comm_path = nx.shortest_path(self.network, source=source_chiplet_id, target=destination_chiplet_id)
                        comm_path = comm_path[:-1]

                        if len(comm_path) > 0:
                            comm_act_latency_this_chiplet = len(comm_path)* activation_pkts_this_chiplet * self.link_params['per_hop_latency']
                            for chiplet_id in comm_path:
                                network_comm_energies[chiplet_id-1] += activation_pkts_this_chiplet * self.link_params['link_width'] * self.link_params['energy_per_bit']
                        
                        comm_act_latency_this_layer = max(comm_act_latency_this_layer, comm_act_latency_this_chiplet)
                
                # weights from io to chiplet
                for next_mapping in dst_mapping:
                    comm_weights_latency_this_chiplet = 0

                    destination_chiplet_id, next_percentage = next_mapping
                    weights_pkts_this_chiplet = weights_pkts * next_percentage / 100

                    min_distance = 10**6
                    for io_chiplet in self.io_chiplets:
                        comm_path = nx.shortest_path(self.network, source=io_chiplet, target=destination_chiplet_id)
                        if len(comm_path) < min_distance:
                            min_distance = len(comm_path)
                            source_chiplet_id = io_chiplet
                    
                    comm_path = nx.shortest_path(self.network, source=source_chiplet_id, target=destination_chiplet_id)
                    comm_path = comm_path[1:]

                    if len(comm_path) > 0:
                        comm_weights_latency_this_chiplet= len(comm_path)* weights_pkts_this_chiplet * self.link_params['per_hop_latency']
                        
                        for chiplet_id in comm_path:
                            network_comm_energies[chiplet_id-1] += weights_pkts_this_chiplet * self.link_params['link_width'] * self.link_params['energy_per_bit']
                        
                    comm_weights_latency_this_layer = max(comm_weights_latency_this_layer, comm_weights_latency_this_chiplet)

            network_comm_times[i] = comm_act_latency_this_layer
            weights_comm_times[i] = comm_weights_latency_this_layer
        
        network_comm_energies = network_comm_energies * self.total_inputs

        compute_comm_times = network_compute_times + network_comm_times

        # total energy and latency
        end_to_end_latency_1_input = np.sum(network_compute_times) + np.sum(network_comm_times)

        end_to_end_latency = (np.max(compute_comm_times))* (self.total_inputs - 1) + end_to_end_latency_1_input + np.sum(weights_comm_times)

        # leakage energy: if crossbar is used, then leakage energy is added for end_to end latency
        leakage_energy = self.network_crossbar_required * end_to_end_latency * self.leakae_power_per_crossbar
        total_energy = np.sum(network_compute_energy) + np.sum(network_comm_energies) + np.sum(leakage_energy)

        self.total_latency = end_to_end_latency
        self.original_total_latency = end_to_end_latency

        self.total_energy = total_energy
        self.total_leakage_energy = np.sum(leakage_energy)

        self.time_remaining = end_to_end_latency

        network_chiplets_energy = network_compute_energy + network_comm_energies + leakage_energy

        self.networks_chiplets_energy = network_chiplets_energy
        self.network_active_energy = network_compute_energy + network_comm_energies

    def update_busy_times_and_power(self):
        """
        For each chiplet, if crossbar required is greater than 0, allocate the crossbar; update the busy times and active power of the chiplet.
        """
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            crossbars_required = self.network_crossbar_required[chiplet_id - 1]

            if crossbars_required > 0:
                remaining_crossbars = crossbars_required
                for tile_num in range(chiplet.tiles_per_chiplet):
                    for crossbar_num in range(chiplet.crossbars_per_tile):
                        if remaining_crossbars > 0:
                            if chiplet.crossbar_available[tile_num][crossbar_num]:
                                # Allocate the crossbar
                                chiplet.crossbar_available[tile_num][crossbar_num] = 0
                                self.active_xbars[chiplet_id - 1][tile_num][crossbar_num] = 1
                                # Update the busy time
                                chiplet.crossbar_busy_times[tile_num][crossbar_num] = self.total_latency
                                remaining_crossbars -= 1
            
            # Update the active power of the chiplet
            chiplet.active_power = chiplet.active_power + (self.networks_chiplets_energy[chiplet_id - 1] / self.original_total_latency)

    def update_remaining_time(self, time_elapsed_s):
        """
        Decrements the busy times of the crossbars associated with this mapping.

        Parameters:
            time_elapsed_s (float): The amount of time to decrement (in seconds).
        """
        self.completed = False
        self.time_remaining -= time_elapsed_s
        if self.time_remaining <= 0:
            self.completed = True

    def stall_this_mapping(self, time_elapsed_s):
        """
        Stall this mapping becuse of temperature violation.

        Parameters:
            time_elapsed_s (float): The amount of time to stall (in seconds).

        """
        self.stalled = True

        self.time_remaining += time_elapsed_s
        self.total_latency += time_elapsed_s
        self.total_energy += (self.total_leakage_energy*time_elapsed_s)/self.original_total_latency

        for chiplet in self.chiplets:
            chiplet.active_power = chiplet.active_power - (self.network_active_energy[chiplet.id - 1] / self.original_total_latency)
            
            # increase the busy time of all crossbars being used in this chiplet, xbars are already allocated
            for tile_num in range(chiplet.tiles_per_chiplet):
                for crossbar_num in range(chiplet.crossbars_per_tile):
                    if self.active_xbars[chiplet.id - 1][tile_num][crossbar_num]:
                        chiplet.crossbar_busy_times[tile_num][crossbar_num] += time_elapsed_s

    def resume_this_mapping(self):
        """
        Resume this mapping because temperature is below threshold.
        """
        self.stalled = False

        for chiplet in self.chiplets:
            chiplet.active_power = chiplet.active_power + (self.network_active_energy[chiplet.id - 1] / self.original_total_latency)

    def reduce_power_chiplets(self):
        """
        Reduces the active power consumption of the chiplets as mapping is complete now.
        """
        if not self.completed:
            # raise error
            if common.DEBUG_RL:
                print("Error: Mapping operation is not complete.")
            exit(1)
        
        if not self.power_reduced:
            self.power_reduced = True

            for chiplet in self.chiplets:
                chiplet_id = chiplet.id
                chiplet.active_power = chiplet.active_power - self.networks_chiplets_energy[chiplet_id - 1] / self.original_total_latency

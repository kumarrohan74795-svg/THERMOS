import numpy as np
from morl import common

def nn_huristic_for_2nd_level_mapping(available_crossbars, allocate_crossbars, allocate_percentage, chiplets_in_this_cluster, distance_matrix, previous_mapping):
    """
    Maps allocated crossbars to the chiplets based on proximity

    Args:
        available_crossbars (np.array): A numpy array where each index represents the number of crossbars available in that chiplet.
        allocate_crossbars int: Number of crossbars to allocate
        allocate_percentage int: Percentage of crossbars to allocate
        chiplets_in_this_cluster np.array: Indexes of chiplets in this cluster
        distance_matrix 2D np.array: A 2D numpy array where each element represents the distance between two chiplets.
        previous_mapping np.array: A list of previous layer's mapping to the chiplets.
    
    Returns:
        available_crossbars (np.array): A numpy array where each index represents the number of crossbars available after allocation.
        list: A list of action variables, where each action corresponds to a chiplet and contains the percentage of crossbars used by that chiplet.
    """
    # mapping cannot fail, because we already checked the available crossbars

    # First layer
    if np.sum(previous_mapping) == 0:
        # Select the chiplet with the maximum available crossbars because all chiplets are same type
        min_distance_io = np.min(distance_matrix[chiplets_in_this_cluster, available_crossbars.shape[0]:], axis=1)
        sorted_chiplets = chiplets_in_this_cluster[np.lexsort((min_distance_io,-available_crossbars[chiplets_in_this_cluster]))]

    else:
        network_shape = distance_matrix.shape[0]
        previous_mapping_shape = previous_mapping.shape[0]
        padded_mapping = np.zeros(network_shape)
        padded_mapping[:previous_mapping_shape] = previous_mapping

        # sort based on distance first
        distance = np.sum(distance_matrix * padded_mapping[:, np.newaxis], axis=0)
        sorted_chiplets = chiplets_in_this_cluster[np.lexsort((-available_crossbars[chiplets_in_this_cluster], distance[chiplets_in_this_cluster]))]
    
    # Allocate the minimum between crossbars_needed and available_crossbars in this chiplet
    total_crossbars = allocate_crossbars
    mapping_this_action = []
    for chiplet in sorted_chiplets:

        if allocate_crossbars <= 0:
            break

        # Check how many crossbars avail in this chiplet
        available_cb = available_crossbars[chiplet]
        if available_cb <= 0:
            continue

        if available_cb >= allocate_crossbars:
            available_crossbars[chiplet] -= allocate_crossbars
            this_chiplet_percentage = allocate_crossbars*allocate_percentage/total_crossbars
            allocate_crossbars = 0
        
        else:
            available_crossbars[chiplet] = 0
            this_chiplet_percentage = available_cb*allocate_percentage/total_crossbars
            allocate_crossbars -= available_cb
        
        mapping_this_action.append([chiplet+1, this_chiplet_percentage])
            
    # error if allocate_crossbars is not zero
    if allocate_crossbars != 0:
        print(f"Error: allocate_crossbars is not zero, {allocate_crossbars}, {mapping_this_action}")
        exit()
    sum_action = np.sum([x[1] for x in mapping_this_action])

    if sum_action + 1 < allocate_percentage:
        print(f"Error: sum_action {sum_action} + 1 < {allocate_percentage}, {mapping_this_action}")
        exit()

    return available_crossbars, mapping_this_action


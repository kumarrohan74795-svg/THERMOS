import math
import numpy as np
import networkx as nx

def compute_output_dims(input_height, input_width, kernel_size, padding, stride):
    """
    Compute the output dimensions after a convolution operation.

    Args:
        input_height (int): Height of the input feature map.
        input_width (int): Width of the input feature map.
        kernel_size (int): Size of the convolution kernel.
        padding (int): Padding applied to the input.
        stride (int): Stride of the convolution.

    Returns:
        tuple: (output_height, output_width)
    """
    output_height = ((input_height + 2 * padding - kernel_size) // stride) + 1
    output_width = ((input_width + 2 * padding - kernel_size) // stride) + 1
    return output_height, output_width

def process_network(network):
    """
    Process each layer of the network and collect layer-specific information.

    Args:
        network (dict): Dictionary containing network structure.

    Returns:
        list: List of layer information list for the network.
        layer_info = [name, layer_type, input_activation, total_macs, num_weights, filter_size]
    """
    conv_layers = network['conv_layers']
    fc_layers = network['fc_layers']
    input_size = network['input_size']
    pooling_layers = network.get('pooling_layers', [])

    # Input image dimensions
    current_input_height = input_size
    current_input_width = input_size

    # List to hold layer information
    network_layers_info = []

    # Process convolutional layers
    for idx, layer in enumerate(conv_layers):
        # Compute output dimensions
        output_height, output_width = compute_output_dims(
            current_input_height, current_input_width,
            layer['kernel_size'], layer['padding'], layer['stride']
        )

        name = layer['name']
        layer_type = 'conv'
        input_activation = current_input_height*current_input_width*layer['in_channels']

        total_macs = layer['out_channels'] * output_height * output_width * (layer['kernel_size'] ** 2) * layer['in_channels']
        num_weights = layer['out_channels'] * layer['in_channels'] * (layer['kernel_size'] ** 2)
        filter_size = layer['in_channels'] * (layer['kernel_size'] ** 2)
        out_channels = layer['out_channels']
        
        # Apply pooling if applicable
        if idx in pooling_layers:
            output_height = output_height // 2
            output_width = output_width // 2

        output_activation = output_height * output_width * out_channels

        # Add layer info to a dict
        layer_info = {
            'name': name,
            'layer_type': layer_type,
            'input_activation': input_activation,
            'total_macs': total_macs,
            'num_weights': num_weights,
            'filter_size': filter_size,
            'out_channels': out_channels,
            'output_activation': output_activation,
        }
        
        # Add to network layers info
        network_layers_info.append(layer_info)

        # Update input dimensions for next layer
        current_input_height, current_input_width = output_height, output_width

    # Process fully connected layers
    for layer in fc_layers:

        name = layer['name']
        layer_type = 'fc'
        input_activation = layer['in_features']
        total_macs = layer['out_features'] * layer['in_features']
        num_weights = layer['out_features'] * layer['in_features']
        in_features = layer['in_features']
        out_features = layer['out_features']
        output_activation = out_features

        # Add layer info to a dict
        layer_info = {
            'name': name,
            'layer_type': layer_type,
            'input_activation': input_activation,
            'total_macs': total_macs,
            'num_weights': num_weights,
            'in_features': in_features,
            'out_features': out_features,
            'output_activation': output_activation,
        }
        
        # Add to network layers info
        network_layers_info.append(layer_info)

    return network_layers_info


def calculate_layer_requirements(layer_info, chiplet):
    """
    Calculate chiplet-specific requirements for the layer.

    Args:
        layer_info (list): Information about the layer. 
        chiplet (Chiplet): Chiplet object with specifications.

    Returns:
        total_crosssbars_layer (int): Total number of crossbars required for the layer.
    """
    # Extract chiplet-specific parameters using dot notation
    CROSSBAR_ROWS = chiplet.crossbar_rows
    CROSSBAR_COLUMNS = chiplet.crossbar_columns
    BITS_PER_CELL = chiplet.bits_per_cell
    BITS_PER_WEIGHT = chiplet.bits_per_weight # 8 all the time

    if layer_info['layer_type'] == 'conv':
        filter_size = layer_info['filter_size'] # filter_size = in_channels * (kernel_size ** 2)
        total_filters = layer_info['out_channels']

        col_required = total_filters * BITS_PER_WEIGHT / BITS_PER_CELL
        row_required = filter_size

        col_crossbars = math.ceil(col_required/CROSSBAR_COLUMNS)
        row_crossbars = math.ceil(row_required/CROSSBAR_ROWS)

        total_crossbars_layer = col_crossbars * row_crossbars

    elif layer_info['layer_type'] == 'fc':
        col_required = layer_info['out_features'] * BITS_PER_WEIGHT/BITS_PER_CELL
        row_required = layer_info['in_features']

        col_crossbars = math.ceil(col_required/CROSSBAR_COLUMNS)
        row_crossbars = math.ceil(row_required/CROSSBAR_ROWS)

        total_crossbars_layer = col_crossbars * row_crossbars
    
    return total_crossbars_layer

def build_chiplet_network(adj_matrix_file, chiplet_mapping, io_chiplets):
    """
    Build the chiplet network topology from the adjacency matrix file and chiplet mapping.

    Args:
        adj_matrix_file (str): Filename of the adjacency matrix.
        chiplet_mapping (dict): Mapping from chiplet numbers to chiplet types.

    Returns:
        networkx.Graph: Graph representing the chiplet network with additional metrics.
    """
    # Load adjacency matrix
    adj_matrix = np.loadtxt(adj_matrix_file)
    num_chiplets = len(chiplet_mapping)

    # Create a graph
    G = nx.Graph()

    # Add nodes with chiplet types and additional metrics
    for chiplet_num in range(num_chiplets):
        chiplet_id = chiplet_num + 1  # Chiplet numbers start from 1
        chiplet_type = chiplet_mapping.get(chiplet_id, 'Unknown')

        # Add node with all attributes
        G.add_node(
            chiplet_id,
            chiplet_type=chiplet_type,
        )

    for io_chiplet in io_chiplets:
        G.add_node(
            io_chiplet,
            chiplet_type='IO',
        )
    
    total_nodes = num_chiplets + len(io_chiplets)

    # Add edges based on adjacency matrix
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i + 1, j + 1)  # Adjust index to start from 1

    return G

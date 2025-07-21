# Chiplet performance parameters (for different chiplet types)
# From CiMLoop tool
# CiMLoop: A Fast and Flexible Simulator for Analog In-Memory Computing Accelerators.

CHIPLET_TYPES = [
    {
        'name': 'Standard',         # Basic Analog
        'tops': 3.06176,               # Terra operations per second
        'energy_per_mac': 873.109504,      # in fJ
        'idle_power_per_crossbar': 0.0001,  # in Watts 
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 2,         # Resolution of the individual crossbarl cells
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 299,
        'threshold_temp' : 330,      # in degree Kelvin
    },
    { 
        'name': 'Shared_ADC',       # Macro A, jia_jssc_2020
        'tops': 3.13344,       
        'energy_per_mac': 303.928319999999,
        'idle_power_per_crossbar': 0.000808,
        'crossbar_rows': 768,
        'crossbar_columns': 768,
        'bits_per_cell': 1,
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 17,
        'threshold_temp' : 358,      # in degree Kelvin
    },
    {
        'name': 'Accumulator',      # Macro C, wan nature 2022
        'tops': 3.510857143,         # Adjusted value
        'energy_per_mac': 222.45888,
        'idle_power_per_crossbar': 0.0001,  # in Watts (example value)
        'crossbar_rows': 256,
        'crossbar_columns': 256,
        'bits_per_cell': 2,
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 150,
        'threshold_temp' : 330,      # in degree Kelvin
    },
    {
        'name': 'ADC_less',         # Macro E, colonnade jssc 2021
        'tops': 0.38656,         # Adjusted value
        'energy_per_mac': 268.704,
        'idle_power_per_crossbar': 0.00122,  # in Watts (example value)
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 1,
        'bits_per_weight': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 151,
        'threshold_temp' : 358,      # in degree Kelvin
    },
    # Add more chiplet types as needed
]

import numpy as np
from morl import common
from params.chiplet_params_new import CHIPLET_TYPES

class Chiplet:
    """
    Represents a single chiplet with specific hardware attributes.
    """

    def __init__(self, chiplet_id, chiplet_type):
        """
        Initializes a Chiplet with given hardware attributes.

        Parameters:
            chiplet_id (int): Unique identifier for the chiplet.
            chiplet_type (str): Name/type of the chiplet.
        """
        self.id = chiplet_id
        self.name = chiplet_type
        chiplet_spec = next((c for c in CHIPLET_TYPES if c['name'] == chiplet_type), None)

        if chiplet_spec is None:
            raise ValueError(f"Chiplet type '{chiplet_type}' not found in CHIPLET_TYPES.")

        # Initialize chiplet specifications
        for key, value in chiplet_spec.items():
            setattr(self, key, value)

        # Bits per weight is always 8
        self.memory_per_crossbar = self.crossbar_rows * self.crossbar_columns * self.bits_per_cell/ self.bits_per_weight

        # Initialize crossbar availability and tracking
        self.initialize_crossbars()

        self.active_power = 0 # Current active power consumption

    def initialize_crossbars(self):
        """
        Initializes the crossbar availability and tracking for the chiplet.
        """
        self.crossbar_busy_times = np.zeros((self.tiles_per_chiplet, self.crossbars_per_tile))
        self.crossbar_available = np.zeros((self.tiles_per_chiplet, self.crossbars_per_tile))

        for tile_num in range(self.tiles_per_chiplet):
            for crossbar_num in range(self.crossbars_per_tile):
                self.crossbar_busy_times[tile_num][crossbar_num] = 0.0  # Remaining busy time (s)
                self.crossbar_available[tile_num][crossbar_num] = 1  # 1 means available

    def get_available_crossbars(self):
        """
        Returns the number of available crossbars in the chiplet.

        Returns:
            int: Number of available crossbars.
        """
        return np.sum(self.crossbar_available == 1).astype(int)

    def get_available_memory(self):
        """
        Returns the number of available memory units in the chiplet.

        Returns:
            int: Number of available memory units.
        """
        return self.get_available_crossbars() * self.memory_per_crossbar
    
    def get_total_memory(self):
        """
        Returns the total amount of weights that the chiplet can hold.

        Returns:
            float: maximum number of weights that can be stored.
        """
        return self.crossbars_per_tile * self.tiles_per_chiplet *  self.memory_per_crossbar
        
    def calculate_current_power(self):
        """
        Calculates the current power state of the chiplet based on the active or idle state of each crossbar.

        Returns:
            float: Total power consumption of the chiplet in Watts.
        """
        # return self.active_power + idle_crossbars * self.idle_power_per_crossbar
        return self.active_power

import numpy as np

class EntireNetwork:
    def __init__(self, network_name, total_inputs):
        self.network_name = network_name
        self.total_inputs = total_inputs
        self.mapping_operations = []

        self.completed = False
        self.last_layer = False
    
        self.waiting_time = 0
        self.total_latency = 0
        self.extra_latency = 0

        self.extra_energy = 0
        self.total_energy = 0

    def add_operation(self, operation, last_layer):
        self.mapping_operations.append(operation)
        self.last_layer = last_layer

        if last_layer:
            # time:
            self.waiting_time = min([operation.waiting_time for operation in self.mapping_operations])
            extra_wait_time = np.array([operation.waiting_time - self.waiting_time for operation in self.mapping_operations])
            latency = np.array([operation.total_latency for operation in self.mapping_operations])
            self.total_latency = np.max(latency + extra_wait_time)

            # energy:
            self.total_energy = np.sum([operation.total_energy for operation in self.mapping_operations])

    def update_remaining_time(self, time):
        for operation in self.mapping_operations:
            # its ok if time goes negative
            operation.update_remaining_time(time)
        
        self.completed = all([operation.completed for operation in self.mapping_operations]) and self.last_layer

        if self.completed:
            # time:
            self.waiting_time = min([operation.waiting_time for operation in self.mapping_operations])
            extra_wait_time = np.array([operation.waiting_time - self.waiting_time for operation in self.mapping_operations])
            latency = np.array([operation.total_latency for operation in self.mapping_operations])
            
            self.extra_latency = np.max(latency + extra_wait_time) - self.total_latency
            self.total_latency = np.max(latency + extra_wait_time)

            # energy:
            self.extra_energy = np.sum([operation.total_energy for operation in self.mapping_operations]) - self.total_energy
            self.total_energy = np.sum([operation.total_energy for operation in self.mapping_operations])

    def reduce_power_chiplets(self):
        for operation in self.mapping_operations:
            if operation.completed:
                operation.reduce_power_chiplets()

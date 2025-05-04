import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
import torch.onnx
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from networks import VectorLyapunovNetwork, CombinedController

# Import from pre_train_model
from pre_train_model.microgrid_simulate import MicrogridSimulator
from pre_train_model.learn_dynamics_control import DynamicsNN, ControllerNN

# Create output directories if they don't exist
os.makedirs('model_weights', exist_ok=True)
os.makedirs('output_figures', exist_ok=True)

class InterconnectedSystem:
    def __init__(self, dynamics_params, connection_matrix):
        """
        dynamics_params: parameters for system dynamics
        connection_matrix: adjacency matrix showing system interconnections
        """
        self.params = dynamics_params
        self.connections = connection_matrix
        
    def next_state(self, states, controls, disturbances):
        """
        Compute next states for all subsystems
        """
        raise NotImplementedError("Implement system-specific dynamics")

class MicrogridFormationDynamics(InterconnectedSystem):
    def __init__(self, dynamics_params, connection_matrix, if_neural_network=True):
        """
        dynamics_params: {
            'dt': timestep,
            'n': number of inverters,
            'omega_star': desired frequency,
            'tau': time constants,
            'eta': droop gains,
            'k': secondary control gains,
            'V': voltage magnitudes,
            'B': line susceptances,
            'P_L': loads,
            'P_star': desired power injections
        }
        connection_matrix: adjacency matrix for inverter connections
        """
        super().__init__(dynamics_params, connection_matrix)
        self.dt = dynamics_params['dt']
        
        # Microgrid parameters
        self.n = dynamics_params.get('n', 3)
        self.omega_star = dynamics_params.get('omega_star', 2 * np.pi * 50)
        self.tau = dynamics_params.get('tau', [1.4895] * self.n)
        self.eta = dynamics_params.get('eta', [6.3509e-4] * self.n)
        self.k = dynamics_params.get('k', [4.9481] * self.n)
        self.V = dynamics_params.get('V', [325.3] * self.n)
        self.B = dynamics_params.get('B', np.zeros((self.n, self.n)))
        self.P_L = dynamics_params.get('P_L', [1260.0] * self.n)
        self.P_star = dynamics_params.get('P_star', [1260.0] * self.n)
        
        self.if_neural_network = if_neural_network
        if self.if_neural_network:
            self.neural_systems = []
            for i in range(self.n):
                if i == 0:
                    model = DynamicsNN(state_dim=3, neighbor_dim=3, control_dim=1)
                    model.load_state_dict(torch.load(f"pre_train_model/dynamics_model_0.pth"))
                elif i == self.n - 1:
                    model = DynamicsNN(state_dim=3, neighbor_dim=3, control_dim=1)
                    model.load_state_dict(torch.load(f"pre_train_model/dynamics_model_2.pth"))
                else:
                    model = DynamicsNN(state_dim=3, neighbor_dim=6, control_dim=1)
                    model.load_state_dict(torch.load(f"pre_train_model/dynamics_model_1.pth"))
                model.eval()
                print(f"Loaded dynamics model for inverter {i}")
                self.neural_systems.append(model)


    
    def compute_power_flow(self, delta):
        """Compute power flow at each node."""
        P_I = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.B[i,j] != 0:
                    P_I[i] += abs(self.B[i,j]) * self.V[i] * self.V[j] * np.sin(delta[i] - delta[j])
        return self.P_L + P_I
    
    def next_state(self, states, control, disturbances, eval=False):
        """
        Compute next states for all inverters
        states: tensor of shape [batch_size, num_inverters, 3] (delta, omega, xi)
        controls: tensor of shape [batch_size, num_inverters]
        disturbances: tensor of shape [batch_size, num_inverters, 3]
        """
        batch_size = states.shape[0]
        num_inverters = states.shape[1]
        next_states = []

        # Other inverters using their respective neural networks
        for i in range(num_inverters):
            if i == 0:
                neighbor_states = states[:, i+1, :].reshape(batch_size, -1)
            elif i == num_inverters - 1:  # Last inverter
                neighbor_states = states[:, i-1, :].reshape(batch_size, -1)
            else:  # Middle inverters
                neighbor_states = states[:, [i-1, i+1], :].reshape(batch_size, -1)
            
            next_states.append(self.neural_systems[i](
                states[:, i, :],
                neighbor_states,
                control[:, i].unsqueeze(-1)
            ))

        # Stack all states together
        return torch.stack(next_states, dim=1)  # [batch_size, num_inverters, 3]

class MicrogridDataModule(pl.LightningDataModule):
    def __init__(self, n_inverters=3, dim=3, batch_size=32, controlled_indices=None):
        super().__init__()
        self.n_inverters = n_inverters
        self.dim = dim  # 3 for (delta, omega, xi)
        self.batch_size = batch_size
        self.controlled_indices = controlled_indices if controlled_indices else []
        # Reference values for microgrid
        self.omega_star = 2 * np.pi * 50  # Nominal frequency (50 Hz)
        
    def setup(self, stage=None):
        # Generate data samples for microgrid states
        num_samples = 7000
        
        # Initialize states tensor with float32 dtype
        # Shape: [num_samples, n_inverters, 3] for (delta, omega, xi)
        states = torch.zeros(num_samples, self.n_inverters, self.dim, dtype=torch.float32)
        
        # First inverter (reference)
        # Phase angle (delta) - uniform distribution around 0
        states[:, 0, 0] = torch.rand(num_samples, dtype=torch.float32) * np.pi/4
        # Frequency (omega) - uniform distribution around nominal frequency
        states[:, 0, 1] = self.omega_star + (torch.rand(num_samples, dtype=torch.float32) - 0.5) * 1.5
        # Controller state (xi) - uniform distribution
        states[:, 0, 2] = 0#torch.rand(num_samples, dtype=torch.float32) * 5

        # For other inverters, generate states with appropriate variations
        for i in range(1, self.n_inverters):
            # Phase angle - uniform variations relative to previous inverter
            states[:, i, 0] = torch.rand(num_samples, dtype=torch.float32) * np.pi/4
            
            # Frequency - uniform distribution around nominal frequency
            states[:, i, 1] = self.omega_star + (torch.rand(num_samples, dtype=torch.float32) - 0.5) * 1.5
            # Controller state - uniform distribution
            states[:, i, 2] = 0#torch.rand(num_samples, dtype=torch.float32) * 5
        
        # Random disturbances for each inverter
        # Shape: [num_samples, n_inverters, 3] for (delta_dist, omega_dist, xi_dist)
        disturbances = torch.zeros(num_samples, self.n_inverters, self.dim, dtype=torch.float32)
        
        # Add small random disturbances to all states using uniform distribution
        #for i in range(self.n_inverters):
        #    disturbances[:, i, 0] = (torch.rand(num_samples, dtype=torch.float32) - 0.5) * 0.1  # Phase angle disturbances
        #    disturbances[:, i, 1] = (torch.rand(num_samples, dtype=torch.float32) - 0.5) * 0.1  # Frequency disturbances
        #    disturbances[:, i, 2] = (torch.rand(num_samples, dtype=torch.float32) - 0.5) * 0.1  # Controller state disturbances
        
        # Split into train and validation
        train_size = int(0.8 * num_samples)
        self.train_states = states
        self.train_disturbances = disturbances
        self.val_states = states[train_size:]
        self.val_disturbances = disturbances[train_size:]

        # Save the data
        torch.save((self.train_states, self.train_disturbances), "data/train_data.pt")
        torch.save((self.val_states, self.val_disturbances), "data/val_data.pt")
    
    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.train_states, self.train_disturbances),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.val_states, self.val_disturbances),
            batch_size=self.batch_size
        )

class StringStabilityTrainer(pl.LightningModule):
    def __init__(self, controller, system, V_net, learning_rate=1e-3, 
                 current_index=None, original_controller=None, shared_lyapunov=False, num_inverters=3):
        super().__init__()
        self.controller = controller
        self.system = system
        self.V_net = V_net
        self.learning_rate = learning_rate
        if current_index is None:
            self.current_index = 0
        else:
            self.current_index = current_index

        print("current_index", self.current_index)

        self.original_controller = original_controller
        self.automatic_optimization = False
        self.shared_lyapunov = shared_lyapunov
        self.num_inverters = num_inverters

    def configure_optimizers(self):
        # parameters of vector lyapunov network and controller
        if self.current_index > 0:
            parameters = list(self.V_net.parameters()) + list(self.controller.parameters())
        else:
            parameters = list(self.V_net.parameters())

        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

        return optimizer
    
    def create_binary_adjacency_matrix(self, connections):
        """
        Convert connections dictionary to binary adjacency matrix for microgrid
        
        Args:
            connections (dict): Dictionary of connections {i: {j: weight}}
        
        Returns:
            torch.Tensor: Binary adjacency matrix where 1 indicates connection exists
        """
        N = len(connections)  # number of inverters
        G = torch.zeros(N, N, device=self.device)
        
        # Convert weighted connections to binary (0/1) connections
        for i in connections:
            for j in connections[i]:
                G[i, j] = 1
        
        return G
    
    def vector_lyapunov_conditions(self, states, disturbances):
        """
        Verify vector Lyapunov conditions for string stability in microgrid
        
        states: [batch_size, num_inverters, 3] (delta, omega, xi)
        disturbances: [batch_size, num_inverters, 3]
        """
        if_fixed_coupling = False
        if if_fixed_coupling:
            coupling_matrix = self.system.connections
        else:
            G = self.create_binary_adjacency_matrix(self.system.connections)
            coupling_matrix = self.V_net.coupling_matrix(G)

        # For microgrid, we consider all inverters except the first one
        inverter_indices = list(range(0, states.shape[1]))
        
        # Calculate error states for each inverter
        batch_size = states.shape[0]
        equilibrium = torch.zeros(batch_size, 3*self.num_inverters, device=states.device)
        for i in range(self.num_inverters):
            equilibrium[:, 3*i+1] = self.system.omega_star
        states_flatten = states.reshape(batch_size, -1)
        states_equilibrium = states_flatten - equilibrium
        
        # Calculate current Lyapunov values
        V_current = self.V_net(states_equilibrium)
        
        # Calculate control inputs
        controls = self.controller(states_equilibrium)
        
        # Calculate next states
        next_states = self.system.next_state(states, controls, disturbances)
        next_states_flatten = next_states.reshape(batch_size, -1)
        next_states_equilibrium = next_states_flatten - equilibrium
        # Calculate next Lyapunov values
        V_next = self.V_net(next_states_equilibrium)
        
        # Calculate V decreases for string stability
        V_decreases = []
        if self.shared_lyapunov:
            iter_num = 3
        else:
            iter_num = states.shape[1]
        for i in range(iter_num):
            decrease = (V_next[:, i] - (1-coupling_matrix[i][i])*V_current[:, i])
            for j in range(states.shape[1]):
                if j != i and j in self.system.connections[i]:
                    decrease -= coupling_matrix[i][j] * V_current[:, j]
            V_decreases.append(decrease)
        
        # Calculate control distance from original controller
        control_dist = torch.tensor(0.0, device=states.device)
        if self.original_controller is not None:
            original_controls = self.original_controller(states_equilibrium)
            control_dist += torch.norm(controls - original_controls, dim=1).mean()/100
        
        return torch.stack(V_decreases), V_current, control_dist
    
    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opts.zero_grad()
        states, disturbances = batch
        
        epsilon = 1e-4
        V_decreases, V_current, control_dist = self.vector_lyapunov_conditions(states, disturbances)
        
        # Loss components for string stability
        loss_decrease = 2000 * torch.relu(V_decreases + epsilon).mean()  # Ensure V decreases
        loss_positive = 4000 * torch.relu(-V_current + epsilon*10).mean()   # Ensure V is positive
        
        # Control loss to maintain similar behavior to original controller
        loss_control = control_dist.mean()
        
        # Total loss
        loss = loss_decrease + loss_positive + loss_control
        
        self.manual_backward(loss)
        opts.step()
        
        # Log losses
        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_decrease', loss_decrease, prog_bar=True)
        self.log('loss_positive', loss_positive, prog_bar=True)
        self.log('loss_control', loss_control, prog_bar=True)
        
        self.scheduler.step()
        return loss
    
    def validation_step(self, batch, batch_idx):
        states, disturbances = batch
        V_decreases, V_current, control_dist = self.vector_lyapunov_conditions(states, disturbances)
        
        # Validation losses (without control loss)
        loss_decrease = 2000 * torch.relu(V_decreases).mean()
        loss_positive = 4000 * torch.relu(-V_current).mean()
        
        val_loss = loss_decrease + loss_positive
        
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss


def train_model(num_inverters=3, controlled_indices=None, state_dims=None, control_dims=None, 
               dynamics_params=None, learning_rate=1e-3, batch_size=32, num_epochs=100, shared_lyapunov=False):
    """
    Train a model for microgrid control using Vector Lyapunov Functions
    
    Args:
        num_inverters: number of inverters in the microgrid
        controlled_indices: indices of inverters to be controlled
        state_dims: dimensions of state variables for each inverter
        control_dims: dimensions of control inputs for each inverter
        dynamics_params: parameters for microgrid dynamics
        learning_rate: learning rate for optimization
        batch_size: batch size for training
        num_epochs: number of training epochs
    """
    if controlled_indices is None:
        # Control all inverters except the first one (reference)
        controlled_indices = list(range(1, num_inverters))
    
    if state_dims is None:
        state_dims = [3] * num_inverters  # Each inverter has (delta, omega, xi)
    
    if control_dims is None:
        control_dims = [1] * num_inverters  # Each inverter has one control input
    
    # Create connection matrix for microgrid
    connection_matrix = {}
    for i in range(num_inverters):
        connection_matrix[i] = {}
        connection_matrix[i][i] = 0.01    # Self connection

        string_stability = True
        if string_stability:
            if i > 0:  # All inverters except reference are connected to their predecessor
                connection_matrix[i][i-1] = 0.01  # Connection to previous inverter
            if i < num_inverters - 1:  # All inverters except last are connected to their successor
                connection_matrix[i][i+1] = 0.01  # Connection to next inverter
        else:
            pass

    # Create system with microgrid dynamics
    system = MicrogridFormationDynamics(dynamics_params, connection_matrix, if_neural_network=True)
    
    # Create binary adjacency matrix for VLF
    G = torch.zeros(len(system.connections), len(system.connections))
    for i in system.connections:
        for j in system.connections[i]:
            G[i, j] = 0.01

    # Initialize data module for microgrid
    data_module = MicrogridDataModule(
        n_inverters=num_inverters,
        dim=3,  # (delta, omega, xi)
        batch_size=batch_size,
        controlled_indices=controlled_indices
    )
    
    # Initialize controller for microgrid
    # Input dimension: current state (3) + neighbor states (3 or 6)
    # Output dimension: control input (1)

    controller = CombinedController(
        input_dim=3,  # Current state + neighbor states
        output_dim=1,  # Control input
        num_inverters=num_inverters,
        shared_lyapunov=shared_lyapunov
    )

    # Create a copy for the original controller
    original_controller = CombinedController(
        input_dim=3,
        output_dim=1,
        num_inverters=num_inverters,
        shared_lyapunov=shared_lyapunov
    )
    
    # Load pre-trained controller weights
    if not shared_lyapunov:
        for i in range(num_inverters):
            if i == 0:
                controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_0.pth"))
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_0.pth"))
            elif i == num_inverters - 1:
                controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_2.pth"))
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_2.pth"))
            else:
                controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_1.pth"))
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_1.pth"))
    else:
        for i in range(3):
            if i == 0:
                controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_0.pth"))
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_0.pth"))
            elif i == 2:
                controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_2.pth"))
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_2.pth"))
            else:
                controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_1.pth"))
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_1.pth"))


    # Create VLF with appropriate input dimension
    # Input dimension: current state (3) + neighbor states (3 or 6)
    V_net = VectorLyapunovNetwork(
        input_dim=3,
        hidden_dim=64,
        G=G,
        shared_lyapunov=shared_lyapunov,
        num_inverters=num_inverters
    )
    
    # Initialize trainer
    trainer = StringStabilityTrainer(
        controller, system, V_net,
        learning_rate=learning_rate, 
        original_controller=original_controller,
        shared_lyapunov=shared_lyapunov,
        num_inverters=num_inverters
    )
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_weights',
        filename='best_microgrid_model',
        save_top_k=1,
        mode='min',
        save_last=False
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )

    # Train the system
    pl_trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        check_val_every_n_epoch=1
    )
    
    pl_trainer.fit(trainer, data_module)
    controller = trainer.controller
    V_net = trainer.V_net
    
    # Compute new coupling matrix using V_net
    new_matrix = V_net.coupling_matrix(G)
    
    # Convert new_matrix back to dictionary form
    new_connections = {}
    for i in range(new_matrix.size(0)):
        new_connections[i] = {}
        for j in range(new_matrix.size(1)):
            val = new_matrix[i, j].item()
            if abs(val) > 0.0:  
                new_connections[i][j] = val

    print("Old connections: ", system.connections)
    print("New connections: ", new_connections)
    system.connections = new_connections
    
    print("Training completed and model saved!")
    return controller, system, V_net

class MicrogridDataModuleRetrain(pl.LightningDataModule):
    def __init__(self, counterexamples, counterexample_ranges, batch_size=32):
        """
        Initialize the data module for microgrid retraining
        
        Args:
            counterexamples: counterexamples for retraining
            counterexample_ranges: ranges for counterexamples
            batch_size: batch size for training
        """
        super().__init__()
        self.counterexamples = counterexamples
        self.counterexample_ranges = counterexample_ranges
        self.batch_size = batch_size
        self.in_train_file = "data/train_data.pt"
        self.out_train_file = "data/train_data.pt"
        self.out_val_file = "data/val_data.pt"

    def setup(self, stage=None):
        # Load existing training data
        old_data_states = torch.load(self.in_train_file)[0]
        old_data_disturbances = torch.load(self.in_train_file)[1]
        old_val_states = torch.load(self.out_val_file)[0]
        old_val_disturbances = torch.load(self.out_val_file)[1]

        # Generate disturbances for counterexamples
        # Shape: [num_samples, num_inverters, 3] for (delta_dist, omega_dist, xi_dist)
        new_data_disturbances = torch.zeros((self.counterexamples.shape[0], self.counterexamples.shape[1], 3))

        # Add counterexamples to training data
        combined_data_states = torch.cat([old_data_states, self.counterexamples], dim=0)
        combined_data_disturbances = torch.cat([old_data_disturbances, new_data_disturbances], dim=0)

        # Add counterexamples to validation data
        combined_val_states = torch.cat([old_val_states, self.counterexamples], dim=0)
        combined_val_disturbances = torch.cat([old_val_disturbances, new_data_disturbances], dim=0)

        # Prepare combined data
        combined_data = (combined_data_states, combined_data_disturbances)
        combined_val_data = (combined_val_states, combined_val_disturbances)

        # Save the new training data
        torch.save(combined_data, self.out_train_file)
        torch.save(combined_val_data, self.out_val_file)

        # Create datasets
        self.train_dataset = TensorDataset(*combined_data)
        self.val_dataset = TensorDataset(*combined_val_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

def add_noise_to_counterexamples(counterexamples):
    """
    Add noise to counterexamples for microgrid system
    
    Args:
        counterexamples: tensor of shape [num_samples, num_inverters, 3] containing (delta, omega, xi)
    
    Returns:
        tensor: expanded counterexamples with added noise
    """
    counter_example_expanded = counterexamples
    for i in range(9):
        noise = (torch.rand_like(counterexamples) - 0.5) * 0.2
        # Add appropriate noise for each state variable
        # Phase angle (delta) - small variations
        noise[:, :, 0] *= 0.1  # [-0.1, 0.1] rad
        
        # Frequency (omega) - variations around nominal frequency
        noise[:, :, 1] *= 0.5  # [-0.5, 0.5] rad/s
        
        # Controller state (xi) - small variations
        noise[:, :, 2] *= 0.0  # [-0.1, 0.1]
        
        # Add the noisy counterexamples
        counter_example_expanded = torch.cat([counter_example_expanded, counterexamples + noise], dim=0)
    
    print("Original counterexamples shape:", counterexamples.shape)
    print("Expanded counterexamples shape:", counter_example_expanded.shape)
    print("Noise ranges:")
    print(f"  Phase angle: [-0.1, 0.1] rad")
    print(f"  Frequency: [-0.5, 0.5] rad/s")
    print(f"  Controller state: [-0.1, 0.1]")

    return counter_example_expanded

def retrain_model(in_system, counterexamples, counterexample_ranges, epoch,
                 in_model, in_controller,
                 learning_rate=1e-4, batch_size=32, index = 0, shared_lyapunov=False, num_inverters=3):
    """
    Retrain the model for microgrid control using Vector Lyapunov Functions
    
    Args:
        in_system: microgrid system dynamics
        counterexamples: counterexamples for retraining
        counterexample_ranges: ranges for counterexamples
        epoch: number of training epochs
        in_model: input vector Lyapunov network
        in_controller: input controller
        learning_rate: learning rate for optimization
        batch_size: batch size for training
        index: current index for training
    """
    V_net = in_model
    controller = in_controller
    system = in_system

    # Create binary adjacency matrix for VLF
    G = torch.zeros(len(system.connections), len(system.connections))
    for i in system.connections:
        for j in system.connections[i]:
            G[i, j] = 1.0

    # Convert counterexamples to tensor and add noise
    counterexamples = torch.tensor(counterexamples)
    counterexamples = add_noise_to_counterexamples(counterexamples)

    # Initialize data module for microgrid retraining
    data_module = MicrogridDataModuleRetrain(counterexamples, counterexample_ranges, batch_size)
    
    # Create a copy for the original controller
    # Input dimension: current state (3) + neighbor states (3 or 6)
    # Output dimension: control input (1)
    original_controller = CombinedController(
        input_dim=3,
        output_dim=1,
        num_inverters=num_inverters,
        shared_lyapunov=shared_lyapunov
    )
     # Load pre-trained controller weights
    if not shared_lyapunov:
        for i in range(num_inverters):
            if i == 0:
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_0.pth"))
            elif i == num_inverters - 1:
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_2.pth"))
            else:
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_1.pth"))
    else:
        for i in range(3):
            if i == 0:
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_0.pth"))
            elif i == 2:
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_2.pth"))
            else:
                original_controller.controllers[i].load_state_dict(torch.load("pre_train_model/control_model_1.pth"))
    # Initialize trainer
    trainer = StringStabilityTrainer(
        controller, system, V_net,
        learning_rate=learning_rate, 
        current_index=index, 
        original_controller=original_controller,
        shared_lyapunov=shared_lyapunov,
        num_inverters=num_inverters
    )

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_weights',
        filename=f'best_microgrid_model',
        save_top_k=1,
        mode='min',
        save_last=False
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )

    # Train the system
    pl_trainer = pl.Trainer(
        max_epochs=epoch,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    pl_trainer.fit(trainer, data_module)
    controller = trainer.controller
    V_net = trainer.V_net
    
    # Compute new coupling matrix using V_net
    new_matrix = V_net.coupling_matrix(G)
    
    # Convert new_matrix back to dictionary form
    new_connections = {}
    for i in range(new_matrix.size(0)):
        new_connections[i] = {}
        for j in range(new_matrix.size(1)):
            val = new_matrix[i, j].item()
            if abs(val) > 0.0:
                new_connections[i][j] = val

    print("Old connections: ", system.connections)
    print("New connections: ", new_connections)
    system.connections = new_connections
    
    print("Training completed and model saved!")
    return controller, system, V_net



if __name__ == "__main__":
    # System parameters
    num_inverters = 3  # Number of inverters in the microgrid
    controlled_indices = list(range(1, num_inverters))  # Control all inverters except the first one
    
    # Dynamics parameters for microgrid
    dynamics_params = {
        'dt': 0.01,  # Time step (s)
        'n': num_inverters,  # Number of inverters
        'omega_star': 2 * np.pi * 50,  # Nominal frequency (50 Hz)
        'tau': [1.4895] * num_inverters,  # Time constants
        'eta': [6.3509e-4] * num_inverters,  # Droop gains
        'k': [4.9481] * num_inverters,  # Secondary control gains
        'V': [325.3] * num_inverters,  # Voltage magnitudes
        'B': np.array([  # Line susceptances
            [0, 0.1, 0],
            [0.1, 0, 0.1],
            [0, 0.1, 0]
        ]),
        'P_L': [1260.0] * num_inverters,  # Load powers
        'P_star': [1260.0] * num_inverters  # Desired power injections
    }
    
    # Training parameters
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 50
    
    # Train the model
    controller, system, V_net = train_model(
        num_inverters=num_inverters,
        controlled_indices=controlled_indices,
        dynamics_params=dynamics_params,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    
    print("Training completed successfully!")
    print(f"Number of inverters: {num_inverters}")
    print(f"Controlled inverters: {controlled_indices}")
    print(f"Nominal frequency: {dynamics_params['omega_star']/(2*np.pi):.2f} Hz")
    print(f"Time step: {dynamics_params['dt']} s")
    print(f"Training epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")




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
from lightning.pytorch.strategies import DDPStrategy
from networks import VectorLyapunovNetwork, CombinedController

# Import from pre_train_model
from pre_train_model.simulate_UAVs import UAVFormation, FormationVisualizer, PDController, LQRController
from pre_train_model.learn_dynamics_control import DynamicsNN, ControllerNN, NNController

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

class UAVFormationDynamics(InterconnectedSystem):
    def __init__(self, dynamics_params, connection_matrix, if_neural_network=True, neural_system=None, train_system=False, device='cuda:0'):
        """
        dynamics_params: {
            'dt': timestep,
            'dim': dimensions (2D or 3D),
            'max_thrust': maximum thrust,
            'min_thrust': minimum thrust,
            'cruise_velocity': cruise velocity vector,
            'delta_ref': reference relative position vector
        }
        connection_matrix: adjacency matrix for UAV connections
        """
        super().__init__(dynamics_params, connection_matrix)
        self.dt = dynamics_params['dt']
        
        # UAV parameters
        self.dim = dynamics_params.get('dim', 3)
        self.max_thrust = dynamics_params.get('max_thrust', 5.0)
        self.min_thrust = dynamics_params.get('min_thrust', -5.0)
        self.cruise_velocity = dynamics_params.get('cruise_velocity', np.zeros(self.dim))
        self.delta_ref = dynamics_params.get('delta_ref', np.array([5.0, 0.0, 0.0]))
        
        # For disturbance
        self.dist_amplitude = dynamics_params.get('dist_amplitude', np.zeros(self.dim))
        self.dist_frequency = dynamics_params.get('dist_frequency', np.zeros(self.dim))
        
        self.if_neural_network = if_neural_network
        if self.if_neural_network:
            # Load pre-trained neural network
            if neural_system is None:
                # Create and load pre-trained network
                self.neural_system = DynamicsNN(state_dim=2*self.dim, action_dim=self.dim).to(device)
                self.neural_system.load_state_dict(torch.load("pre_train_model/dynamics_model.pth"))
                self.neural_system.eval()
            else:
                self.neural_system = neural_system
            
            if train_system:
                self.train_neural_uav_dynamics()
    
    
    def next_state(self, states, control, disturbances, eval=False):
        """
        Compute next states for all UAVs in formation
        states: tensor of shape [batch_size, num_uavs, 2*dim] (position and velocity)
        controls: list of tensors of shape [batch_size, dim] for controlled UAVs (None for uncontrolled)
        disturbances: tensor of shape [batch_size, num_uavs, dim]
        """
        
        batch_size = states.shape[0]
        num_uavs = states.shape[1]
        next_states = []

        # Extract positions and velocities
        p = states[:, :, :self.dim]  # positions
        v = states[:, :, self.dim:]  # velocities

        leader_next_p = p[:, 0, :] + v[:, 0, :] * self.dt
        leader_next_v = v[:, 0, :] #+ disturbances[:, 0, :] * self.dt
        next_states.append(torch.cat([leader_next_p, leader_next_v], dim=-1))

        for i in range(1,num_uavs):
            #print("states[0,i,:]", states[0,i,:])
            #print("control[0,(i-1)*3:(i)*3]", control[0,(i-1)*3:(i)*3])
            next_states.append(self.neural_system(states[:,i,:], control[:,(i-1)*3:(i)*3]))
            #print("next_states[-1][0,:]", next_states[-1][0,:])


        # Stack all states together
        return torch.stack(next_states, dim=1)  # [batch_size, num_uavs, 2*dim]
    

class UAVDataModule(pl.LightningDataModule):
    def __init__(self, n_uavs=2, dim=3, batch_size=32, controlled_indices=None):
        super().__init__()
        self.n_uavs = n_uavs
        self.dim = dim
        self.batch_size = batch_size
        self.controlled_indices = controlled_indices if controlled_indices else []
        # Fixed reference spacing between UAVs (matches system delta_ref)
        self.delta_ref = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32)
        
    def setup(self, stage=None):
        # Generate data samples focused on formation errors
        num_samples = 5000
        
        # Initialize states tensor with float32 dtype
        states = torch.zeros(num_samples, self.n_uavs, 2*self.dim, dtype=torch.float32)
        
        # Fix leader position and velocity with small variations
        # Leader flies primarily along x-axis at cruise velocity
        states[:, 0, 0] = 0.0 #torch.randn(num_samples, dtype=torch.float32) * 2.0  # x position centered at 0
        # Leader velocity - primarily in x direction
        states[:, 0, self.dim] = 5.0 

        # For follower UAVs, generate positions based on desired spacing plus errors
        for i in range(1, self.n_uavs):
            # Position error distribution - wider for training diversity
            pos_error = torch.rand(num_samples, self.dim, dtype=torch.float32)-0.5
            pos_error[:, 0] *= 8.0  # Larger x-direction errors (range ~ ±6m)
            pos_error[:, 1] = 0  # Moderate y-direction errors (range ~ ±3m)
            pos_error[:, 2] = 0
            
            # Velocity error distribution
            vel_error = torch.rand(num_samples, self.dim, dtype=torch.float32)-0.5
            vel_error[:, 0] *= 8.0  # x-velocity error (range ~ ±3m/s)
            vel_error[:, 1] = 0
            vel_error[:, 2] = 0
            
            # Follower position = predecessor position - desired spacing + error
            states[:, i, :self.dim] = states[:, i-1, :self.dim] - self.delta_ref + pos_error
            
            # Follower velocity = predecessor velocity + error
            states[:, i, self.dim:] = states[:, i-1, self.dim:] + vel_error
        
        # Random disturbances (primarily affecting the leader)
        disturbances = torch.zeros(num_samples, self.n_uavs, self.dim, dtype=torch.float32)
        
        # Leader disturbances - simulate wind or other external forces
        #disturbances[:, 0, 0] = torch.randn(num_samples, dtype=torch.float32) * 0.3  # x-direction
        #disturbances[:, 0, 1] = torch.randn(num_samples, dtype=torch.float32) * 0.5  # y-direction (stronger crosswind)
        #disturbances[:, 0, 2] = torch.randn(num_samples, dtype=torch.float32) * 0.4  # z-direction
        
        # Follower disturbances - typically smaller due to wake effects
        #for i in range(1, self.n_uavs):
        #    disturbances[:, i, :] = torch.randn(num_samples, self.dim, dtype=torch.float32) * 0.2
        
        # Split into train and validation
        train_size = int(0.8 * num_samples)
        self.train_states = states
        self.train_disturbances = disturbances
        self.val_states = states[train_size:]
        self.val_disturbances = disturbances[train_size:]

        # save the data
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
                 current_index=None, original_controller=None, device=None, mode=0):
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
        self._device = device
        self.mode = mode
        #self.save_hyperparameters(ignore=['controller', 'system', 'original_controller'])

    def configure_optimizers(self):
        # parameters of vector lyapunov network and controller

        if self.current_index > 0 or self.mode == 0 or self.mode == 1:
            parameters = list(self.V_net.parameters()) + list(self.controller.parameters())
        else:
            parameters = list(self.V_net.parameters()) # + list(self.controller.parameters())

        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

        return optimizer
    
    def create_binary_adjacency_matrix(self, connections):
        """
        Convert connections dictionary to binary adjacency matrix
        
        Args:
            connections (dict): Dictionary of connections {i: {j: weight}}
        
        Returns:
            torch.Tensor: Binary adjacency matrix where 1 indicates connection exists
        """
        N = len(connections)  # number of vehicles
        G = torch.zeros(N, N, device=self.device)
        
        # Convert weighted connections to binary (0/1) connections
        for i in connections:
            for j in connections[i]:
                G[i, j] = 1
        
        return G
    
    def vector_lyapunov_conditions(self,states,disturbances):
        """
        Verify vector Lyapunov conditions for string stability
        
        states: [batch_size, num_UAVs, 6]
        disturbances: [batch_size, num_UAVs, 3]
        """
        if_fixed_coupling = False
        if if_fixed_coupling:
            coupling_matrix = self.system.connections
        else:
            G = self.create_binary_adjacency_matrix(self.system.connections)
            coupling_matrix = self.V_net.coupling_matrix(G)

        UAV_index = [1,2]
        delta_ref_tensor = torch.tensor(self.system.delta_ref, device=states.device)

        # calculate error state for each UAV, and then concatenate them
        p_error = []
        v_error = []
        for i in range(len(UAV_index)):
            state_i = states[:,UAV_index[i],:]
            state_i_preceding = states[:,UAV_index[i]-1,:]
            
            delta_ref_expanded = delta_ref_tensor.unsqueeze(0).expand(state_i.shape[0], -1).to(torch.float32)
            #print("state_i_preceding", state_i_preceding)
            #print("state_i", state_i)
            p_error_i = state_i_preceding[:, :self.system.dim] - state_i[:, :self.system.dim] - delta_ref_expanded
            v_error_i = state_i_preceding[:, self.system.dim:] - state_i[:, self.system.dim:]
            p_error.append(p_error_i)
            v_error.append(v_error_i)

        p_error = torch.cat(p_error, dim=-1)
        v_error = torch.cat(v_error, dim=-1)
        error_state = torch.cat([p_error, v_error], dim=-1)
        #print("error_state", error_state)
        V_current = self.V_net(error_state)
        controls = self.controller(error_state)
        next_states = self.system.next_state(states,controls,disturbances)  

        error_state_next = []
        for i in range(len(UAV_index)):
            state_next_i = next_states[:,UAV_index[i],:]
            state_i_preceding = next_states[:,UAV_index[i]-1,:]
            #print("state_i[0,:]", states[0,UAV_index[i],:])
            #print("state_i_preceding[0,:]", states[0,UAV_index[i]-1,:])
            #print("state_next_i[0,:]", state_next_i[0,:])
            #print("state_i_preceding[0,:]", state_i_preceding[0,:])
            p_error_next_i = state_i_preceding[:, :self.system.dim] - state_next_i[:, :self.system.dim] - delta_ref_expanded
            v_error_next_i = state_i_preceding[:, self.system.dim:] - state_next_i[:, self.system.dim:]
            error_state_next.append(torch.cat([p_error_next_i, v_error_next_i], dim=-1))    
        error_state_next = torch.cat(error_state_next, dim=-1)

        V_next = self.V_net(error_state_next)
        Loss_A_list = []
        V_decreases = []

        for i in range(1,states.shape[1]):
            if self.mode == 0:
                decrease = (V_next[:,i-1] - (1-coupling_matrix[i][i])*V_current[:,i-1])
                for j in range(1,states.shape[1]):
                    if j != i and j in self.system.connections[i]:
                        decrease -= coupling_matrix[i][j] * V_current[:,j-1]
                V_decreases.append(decrease)
            elif self.mode == 1:
                Loss_A = V_current[:,i-1]
                max_other_V = None
                for j in range(1, states.shape[1]):
                    if j != i:
                        if max_other_V == None:
                            max_other_V = V_current[:,j-1]
                        else:
                            max_other_V = torch.max(max_other_V, V_current[:,j-1])
                Loss_A = Loss_A - 0.1*max_other_V

                Loss_B = V_next[:,i-1] - V_current[:,i-1] + 0.01*V_current[:,i-1]

                V_decreases.append(Loss_B)
                Loss_A_list.append(Loss_A)
        control_dist = torch.tensor(0.0, device=states.device)
        control_dist += torch.norm(controls - self.original_controller(error_state), dim=1).mean()
        
        if self.mode == 0:
            return torch.stack(V_decreases), V_current, control_dist
        elif self.mode == 1:
            return torch.stack(V_decreases), torch.stack(Loss_A_list), control_dist

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opts.zero_grad()
        states, disturbances = batch
        #print("states", states)
        epsilon = 1e-5
        V_decreases, V_current, control_dist = self.vector_lyapunov_conditions(states, disturbances)
        loss_decrease = 2000 * torch.relu(V_decreases + epsilon).mean()
        loss_positive = 200 * torch.relu(-V_current + epsilon).mean()

        loss_control = control_dist.mean()
        loss = loss_decrease + loss_positive + loss_control
        self.manual_backward(loss)
        opts.step()
        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_decrease', loss_decrease, prog_bar=True)
        self.log('loss_positive', loss_positive, prog_bar=True)

        self.scheduler.step()
        return loss
    
    def validation_step(self, batch, batch_idx):
        states, disturbances = batch
        V_decreases, V_current, control_dist = self.vector_lyapunov_conditions(states, disturbances)
        loss_decrease = 2000*torch.relu(V_decreases).mean()
        loss_positive = 1000*torch.relu(-V_current).mean()

        val_loss = loss_decrease + loss_positive

        self.log('val_loss', val_loss, prog_bar=True)

        return val_loss


def train_model(num_uavs=3, controlled_indices=None, state_dims=None, control_dims=None, 
               dynamics_params=None, learning_rate=1e-3, batch_size=32, num_epochs=100, device=None, mode=0):
    """
    Train a model for UAV formation control using Vector Lyapunov Functions
    """
    if controlled_indices is None:
        # Control all follower UAVs (indices 1 through num_uavs-1)
        controlled_indices = list(range(1, num_uavs))
    
    if state_dims is None:
        dim = dynamics_params.get('dim', 3)
        state_dims = [2*dim] * num_uavs  # Each UAV has position and velocity
    
    if control_dims is None:
        dim = dynamics_params.get('dim', 3)
        control_dims = [dim] * num_uavs  # Each UAV has control in each dimension
    
    # Create connection matrix
    connection_matrix = {}
    for i in range(num_uavs):
        connection_matrix[i] = {}
        if i > 0:  # All UAVs except leader are connected to their predecessor
            connection_matrix[i][i-1] = 0.01
            connection_matrix[i][i] = 0.01

    # Create system
    system = UAVFormationDynamics(dynamics_params, connection_matrix, device=device)
    
    G = torch.zeros(len(system.connections), len(system.connections)).to(device)
    for i in system.connections:
        for j in system.connections[i]:
            G[i, j] = 0.01

    # Initialize data module
    data_module = UAVDataModule(
        n_uavs=num_uavs,
        dim=dynamics_params.get('dim', 3),
        batch_size=batch_size,
        controlled_indices=controlled_indices
    )
    
    # For each UAV, either load a pre-trained controller or use None
    controller = CombinedController(input_dim=2*system.dim, output_dim=system.dim).to(device)
    controller.controller_1.load_state_dict(torch.load("pre_train_model/controller_model.pth"))
    controller.controller_2.load_state_dict(torch.load("pre_train_model/controller_model.pth"))

    # Create a copy for the original controller
    original_controller = CombinedController(input_dim=2*system.dim, output_dim=system.dim).to(device)
    original_controller.controller_1.load_state_dict(torch.load("pre_train_model/controller_model.pth"))
    original_controller.controller_2.load_state_dict(torch.load("pre_train_model/controller_model.pth"))

    # Error state dimension: position error (dim) + velocity error (dim)
    error_dim = 2 * system.dim
    # Create VLF with appropriate input dimension
    V_net = VectorLyapunovNetwork(input_dim=error_dim, hidden_dim=64, G=G).to(device)
    
    # Initialize trainer with Vector Lyapunov Functions
    trainer = StringStabilityTrainer(
        controller, system, V_net,
        learning_rate=learning_rate, 
        original_controller=original_controller,
        device=device,
        mode=mode
    )
    
    # Setup checkpointing
    if mode == 0:
        file_name = 'best_uav_model'
    elif mode == 1:
        file_name = 'best_uav_model_mode_ISS'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_weights',
        filename=file_name,
        save_top_k=1,
        mode='min',
        save_last=False
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",   # 需要监控的指标（在你的 validation_step 中定义的，比如 val_loss）
        patience=10,           # 在指标无法继续改善的情况下，最多等待多少个验证周期
        verbose=True,         # 是否在终端输出
        mode="min"            # "min" 表示监控的指标越小越好（常见于 loss），"max" 表示指标越大越好（常见于准确率、F1 等）
    )

    # Train the system
    pl_trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        devices=1,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True)
        #enable_progress_bar=True,
        #log_every_n_steps=1
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

class UAVDataModuleRetrain(pl.LightningDataModule):
    def __init__(self, counterexamples, counterexample_ranges, batch_size=32):
        super().__init__()
        self.counterexamples = counterexamples
        self.counterexample_ranges = counterexample_ranges
        self.batch_size = batch_size
        self.in_train_file = "data/train_data.pt"
        self.out_train_file = "data/train_data.pt"
        self.out_val_file = "data/val_data.pt"

    def setup(self, stage=None):
        # 加载原有训练数据
        old_data_states = torch.load(self.in_train_file)[0]
        old_data_disturbances = torch.load(self.in_train_file)[1]
        old_val_states = torch.load(self.out_val_file)[0]
        old_val_disturbances = torch.load(self.out_val_file)[1]

        # generate new x_stars for counterexamples, which is of same size as counterexamples
        new_data_disturbances = torch.zeros((self.counterexamples.shape[0],3,3))

        # 添加反例数据
        combined_data_states = torch.cat([old_data_states, self.counterexamples], dim=0)
        #print("old_data_disturbances", old_data_disturbances.shape)
        #print("new_data_disturbances", new_data_disturbances.shape)
        combined_data_disturbances = torch.cat([old_data_disturbances, new_data_disturbances], dim=0)

        combined_val_states = torch.cat([old_val_states, self.counterexamples], dim=0)
        combined_val_disturbances = torch.cat([old_val_disturbances, new_data_disturbances], dim=0)

        # add some data augementation for original counterexamples
        combined_data = (combined_data_states, combined_data_disturbances)
        combined_val_data = (combined_val_states, combined_val_disturbances)

        # 保存新的训练数据
        torch.save(combined_data, self.out_train_file)
        torch.save(combined_val_data, self.out_val_file)

        # 创建数据集
        self.train_dataset = TensorDataset(*combined_data)
        self.val_dataset = TensorDataset(*combined_val_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

def add_noise_to_counterexamples(counterexamples):
    """
    Add noise to counterexamples
    """
    counter_example_expanded = counterexamples
    for i in range(19):
        noise = (torch.rand_like(counterexamples) - 0.5) * 0.2
        noise[:, 0, :] = 0.0  # No noise on first UAV
        # no noise on pos y z and vel y z
        noise[:, :, 1] = 0.0
        noise[:, :, 2] = 0.0
        noise[:, :, 4] = 0.0
        noise[:, :, 5] = 0.0
        counter_example_expanded = torch.cat([counter_example_expanded, counterexamples + noise], dim=0)
    print("original counterexamples: ", counterexamples.shape)
    print("expanded counterexamples: ", counter_example_expanded.shape)

    return counter_example_expanded

def retrain_model(in_system, counterexamples, counterexample_ranges, epoch,
                 in_model, in_controller,
                 learning_rate=1e-4, batch_size=32, index = 0):
    """
    Retrain the model for UAV formation control using Vector Lyapunov Functions
    """
    V_net = in_model
    controller = in_controller
    system = in_system

    G = torch.zeros(len(system.connections), len(system.connections))
    for i in system.connections:
        for j in system.connections[i]:
            G[i, j] = 1.0

    counterexamples = torch.tensor(counterexamples)
    counterexamples = add_noise_to_counterexamples(counterexamples)

    data_module = UAVDataModuleRetrain(counterexamples, counterexample_ranges, batch_size)
    # Create a copy for the original controller
    original_controller = CombinedController(input_dim=2*system.dim, output_dim=system.dim)
    original_controller.controller_1.load_state_dict(torch.load("pre_train_model/controller_model.pth"))
    original_controller.controller_2.load_state_dict(torch.load("pre_train_model/controller_model.pth"))

    trainer = StringStabilityTrainer(
        controller, system, V_net,
        learning_rate=learning_rate, 
        current_index=index, 
        original_controller=original_controller)

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_weights',
        filename=f'best_uav_model',
        save_top_k=1,
        mode='min',
        save_last=False
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",   # 需要监控的指标（在你的 validation_step 中定义的，比如 val_loss）
        patience=10,           # 在指标无法继续改善的情况下，最多等待多少个验证周期
        verbose=True,         # 是否在终端输出
        mode="min"            # "min" 表示监控的指标越小越好（常见于 loss），"max" 表示指标越大越好（常见于准确率、F1 等）
    )

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

def test_formation_with_neural_controller():
    """
    Test a UAV formation with the neural network controller
    """
    print("Testing UAV formation with neural network controller...")
    
    # Load pre-trained controller
    controller = NNController("pre_train_model/controller_model.pth")
    
    # Create UAV formation with neural controller
    formation = UAVFormation(n=2, dim=3, controller=controller)
    
    # Run simulation
    formation.run_simulation()
    
    # Visualize results
    visualizer = FormationVisualizer(formation)
    visualizer.plot_complete_trajectory(controller_name="Neural Network")
    ani = visualizer.animate_formation(controller_name="Neural Network")
    
    return formation, visualizer, ani

if __name__ == "__main__":
    # System parameters
    num_uavs = 3
    # Control all follower UAVs (indices 1 through 4)
    controlled_indices = list(range(1, num_uavs))
    dim = 3  # 3D space
    
    # Dynamics parameters
    dynamics_params = {
        'dt': 0.1,
        'dim': dim,
        'max_thrust': 5.0,
        'min_thrust': -5.0,
        'cruise_velocity': np.array([5.0, 0.0, 0.0]),  # Mainly flying along X-axis
        'delta_ref': np.array([10.0, 0.0, 0.0]),  # Spacing along X-axis
        'dist_amplitude': np.array([0.2, 0.5, 0.3]),  # Disturbance amplitude
        'dist_frequency': np.array([0.5, 0.3, 0.4])  # Disturbance frequency
    }

    # Training or just testing with pre-trained models
    do_training = True
    
    if do_training:
        # Train the model
        controllers, system, v_nets = train_model(
            num_uavs=num_uavs,
            controlled_indices=controlled_indices,
            dynamics_params=dynamics_params,
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=50
        )
    else:
        # Just test with pre-trained models
        formation, visualizer, ani = test_formation_with_neural_controller()




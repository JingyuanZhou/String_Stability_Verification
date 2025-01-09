import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint

class VectorLyapunovNetwork_with_slice(nn.Module):
    def __init__(self, state_dims, hidden_dim=30):
        """
        Initialize vector Lyapunov function for each subsystem
        state_dims: list of state dimensions for each subsystem
        Note: First vehicle (leading) doesn't need Lyapunov function
        """
        super().__init__()
        # Skip the first vehicle (leading)
        self.lyapunov_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for dim in state_dims[1:]  # Skip first vehicle
        ])
        
        # Initialize R matrices for each subsystem (skip leading vehicle)
        self.R_matrices = nn.ParameterList([
            self._init_R_matrix(dim) for dim in state_dims[1:]
        ])
        
        self.num_vehicles = len(state_dims)
        
    def _init_R_matrix(self, dim):
        """Initialize R matrix with SVD parameterization"""
        U = torch.randn(dim, dim)
        U, _ = torch.linalg.qr(U)  # Orthonormal U
        V = torch.randn(dim, dim)
        V, _ = torch.linalg.qr(V)  # Orthonormal V
        sigma = torch.ones(dim)  # Initial Σ
        r = nn.Parameter(torch.randn(dim))  # Learnable r parameters
        
        return nn.Parameter(U @ (torch.diag(sigma + r**2)) @ V.T)
    
    def forward(self, states, x_stars):
        """
        Compute vector Lyapunov function values for all subsystems except leading vehicle
        states: list of state tensors for each subsystem
        x_stars: list of equilibrium points for each subsystem
        """
        V_values = []
        # Skip the leading vehicle (i starts from 1)
        for i in range(1, self.num_vehicles): #states.shape[1]
            state = states[:,i,:]
            x_star = x_stars[:,i,:]
            net = self.lyapunov_nets[i-1]  # Adjust index since we skipped first vehicle
            R = self.R_matrices[i-1]
            phi_V = net(state)
            phi_V_star = net(x_star)
            state_diff = state - x_star
            R_term = torch.norm(torch.matmul(state_diff, R.T), p=1, dim=1)
            V_i = phi_V - phi_V_star + R_term
            V_values.append(V_i)
            
        return torch.stack(V_values)

class VectorLyapunovNetwork_general(nn.Module):
    def __init__(self, state_dim, hidden_dim=30):
        super().__init__()
        self.num_vehicles = len(state_dim)
        one_state_dim = state_dim[0]
        self.all_state_dim = sum(state_dim)
        self.network = nn.Sequential(
            nn.Linear(self.all_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_vehicles-1)
        )

        # Initialize R matrices for each subsystem (skip leading vehicle)
        self.R_matrices = nn.ParameterList([
            self._init_R_matrix(dim) for dim in state_dim[1:]
        ])

    def _init_R_matrix(self, dim):
        """Initialize R matrix with SVD parameterization"""
        U = torch.randn(dim, dim)
        U, _ = torch.linalg.qr(U)  # Orthonormal U
        V = torch.randn(dim, dim)
        V, _ = torch.linalg.qr(V)  # Orthonormal V
        sigma = torch.ones(dim)  # Initial Σ
        r = nn.Parameter(torch.randn(dim))  # Learnable r parameters
        
        return nn.Parameter(U @ (torch.diag(sigma + r**2)) @ V.T)
        
    def forward(self, x, x_star):
        """
        Compute Lyapunov function value
        """
        x = x.reshape(-1, self.all_state_dim)
        phi_V = self.network(x)
        V = phi_V
        return V
    
class VectorLyapunovNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=30):
        super(VectorLyapunovNetwork, self).__init__()
        self.num_vehicles = len(state_dim)
        self.one_state_dim = state_dim[0]
        self.all_state_dim = sum(state_dim)

        self.network_1 = nn.Sequential(
            nn.Linear(self.one_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.network_2 = nn.Sequential(
            nn.Linear(self.one_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


        W1 = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        W2 = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        W_star = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        
        W1[self.one_state_dim, 0] = 1
        W1[self.one_state_dim+1, 1] = 1

        W2[2*self.one_state_dim, 0] = 1
        W2[2*self.one_state_dim+1, 1] = 1
        W_star[0:self.one_state_dim, torch.arange(self.one_state_dim)] = 1

        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)
        self.register_buffer('W_star', W_star)

    def forward(self, x, x_star):
        """
        计算 Lyapunov 函数值，不使用任何切片或 gather 操作。

        参数:
        - x (Tensor): 输入张量，形状为 [batch_size, all_state_dim]
        - x_star (Tensor): 参考状态张量，形状为 [batch_size, num_star, one_state_dim]

        返回:
        - V (Tensor): Lyapunov 函数值，形状为 [batch_size, 2]
        """

        x = x.view(-1, self.all_state_dim)  
        x_star = x_star.view(-1, self.all_state_dim)


        x1 = torch.matmul(x, self.W1)
        x2 = torch.matmul(x, self.W2)
        x_star_1 = torch.matmul(x_star, self.W_star)
        x_star_2 = torch.matmul(x_star, self.W_star)

        # calculation of Lyapunov function
        V_1 = self.network_1(x1)# - self.network_1(x_star_1)
        V_2 = self.network_2(x2)# - self.network_2(x_star_2)

        V = torch.cat([V_1, V_2], dim=1)
        return V

class NetworkController(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, control_dim)
        )
        self.state_dim = state_dim
        self.W_change_state_position = torch.zeros(self.state_dim, self.state_dim, requires_grad=False)
        #index from 0 1 2 3 4 5 to 1 3 5 0 2 4
        self.W_change_state_position[1, 0] = 1
        self.W_change_state_position[3, 1] = 1
        self.W_change_state_position[5, 2] = 1
        self.W_change_state_position[0, 3] = 1
        self.W_change_state_position[2, 4] = 1
        self.W_change_state_position[4, 5] = 1
        
    def forward(self, x, x_star, u_star, u_bounds):
        """
        Compute control input with clamping
        """
        u_min, u_max = u_bounds
        x = x.reshape(-1, self.state_dim)
        x = x @ self.W_change_state_position
        x_star = x_star.reshape(-1, self.state_dim)
        x_star = x_star @ self.W_change_state_position
        phi_pi = self.network(x)
        phi_pi_star = self.network(x_star)
        u = phi_pi#torch.clamp(phi_pi, u_min, u_max)# - phi_pi_star + u_star
        return u

class system_network(nn.Module):
    def __init__(self, state_dim, hidden_dim=30):
        super().__init__()
        self.state_dim = state_dim

        # network with state and control input as input
        self.network = nn.Sequential(
            nn.Linear(state_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Compute next state
        """
        x = x.reshape(-1, self.state_dim)
        a = self.network(x)
        return a

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

class PlatoonDynamics(InterconnectedSystem):
    def __init__(self, dynamics_params, connection_matrix, if_neural_network=False, neural_system=None,train_system=False):
        """
        dynamics_params: {
            'dt': timestep,
            'alpha': speed adaptation coefficient,
            'beta': relative speed adaptation coefficient,
            'v_max': maximum speed,
            's_st': minimum spacing,
            's_go': maximum spacing,
            'a_max': maximum acceleration,
            'a_min': minimum acceleration,
            'desired_spacing': desired inter-vehicle spacing
        }
        connection_matrix: adjacency matrix for vehicle connections
        """
        super().__init__(dynamics_params, connection_matrix)
        self.dt = dynamics_params['dt']
        
        # OVM parameters
        self.alpha = dynamics_params.get('alpha', 0.6)
        self.beta = dynamics_params.get('beta', 0.9)
        self.v_max = dynamics_params.get('v_max', 30.0)
        self.s_st = dynamics_params.get('s_st', 5.0)
        self.s_go = dynamics_params.get('s_go', 35.0)
        self.a_max = dynamics_params.get('a_max', 7.0)
        self.a_min = dynamics_params.get('a_min', -7.0)
        self.if_neural_network = if_neural_network
        if self.if_neural_network:
            self.neural_system = neural_system
            if train_system:
                self.train_neural_cf_dynamics()
            else:
                self.neural_system.load_state_dict(torch.load("model_weights/neural_dynamics.pth"))
        
    def _compute_hdv_acceleration(self, state_i, state_ahead, eval):
        """Compute HDV acceleration using OVM model"""

        # Extract states
        spacing_i = state_i[..., 0]
        vel_i = state_i[..., 1]
        vel_ahead = state_ahead[..., 1]

        if self.if_neural_network and not eval:
            vel_ahead = state_ahead[..., 1].unsqueeze(1)

            all_states = torch.cat([state_i, vel_ahead], dim=1)
            acc = self.neural_system(all_states)
            acc = torch.clamp(acc, self.a_min, self.a_max)
            return acc
        
        # Calculate desired velocity based on spacing
        cal_D = torch.clamp(spacing_i, self.s_st, self.s_go)
        v_d = self.v_max/2 * (1 - torch.cos(torch.pi * (cal_D - self.s_st)/(self.s_go - self.s_st)))
        
        # Compute acceleration using OVM
        dv = vel_ahead - vel_i
        acc = self.alpha * (v_d - vel_i) + self.beta * dv
        
        # Clamp acceleration
        acc = torch.clamp(acc, self.a_min, self.a_max)
        
        return acc
    
    def next_state(self, states, controls, disturbances, eval = True):
        """
        Compute next states for all vehicles in platoon
        states: tensor of shape [batch_size, num_vehicles, 2]
        controls: list of tensors of shape [batch_size, 1] for CAVs (None for HDVs)
        disturbances: tensor of shape [batch_size, num_vehicles]
        """
        
        batch_size = states.shape[0]
        next_states = []

        # Leading vehicle dynamics (index 0)
        lead_spacing = states[:,0,0]
        lead_vel = states[:,0,1] + disturbances[:,0] * self.dt
        next_states.append(torch.stack([lead_spacing, lead_vel], dim=-1))
        
        # Following vehicles
        for i in range(1, states.shape[1]):
            spacing_i, vel_i = states[:,i,0], states[:,i,1]
            vel_preceding = states[:,i-1,1]
            
            if controls[i] is not None:  # CAV
                acc_i = controls[i]  # [batch_size, 1]
            else:  # HDV
                acc_i = self._compute_hdv_acceleration(states[:,i,:], states[:,i-1,:], eval)
                if not self.if_neural_network:
                    acc_i = acc_i.unsqueeze(-1)  # [batch_size, 1]
            
            next_spacing = spacing_i + (vel_preceding - vel_i) * self.dt
            next_vel = vel_i + acc_i.squeeze(-1) * self.dt
            
            next_states.append(torch.stack([next_spacing, next_vel], dim=-1))
        
        # Stack all states together
        return torch.stack(next_states, dim=1)  # [batch_size, num_vehicles, 2]
    

    def train_neural_cf_dynamics(self, num_epochs=100, learning_rate=1e-4, batch_size=32):
        """
        Train neural network for system dynamics using only Monte Carlo sampling
        """
        mc_size = 30000
        state_dim = 2

        mc_spacing = torch.rand(mc_size) * 20.0 + 5
        mc_vel = torch.rand(mc_size) * 15.0 + 5
        mc_states = torch.stack([mc_spacing, mc_vel], dim=1)
        spacing_ahead = torch.rand(mc_size) * 20.0 + 5
        vel_ahead = torch.rand(mc_size) * 15.0 + 5
        states_ahead = torch.stack([spacing_ahead, vel_ahead], dim=1)
        

        mc_next_acc = self._compute_hdv_acceleration(mc_states, states_ahead, eval = True)

        train_loader = DataLoader(
            TensorDataset(mc_states, states_ahead, mc_next_acc), 
            batch_size=batch_size, 
            shuffle=True
        )

        optimizer = torch.optim.Adam(self.neural_system.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            self.neural_system.train()
            train_loss = 0.0
            for state,preceding_state,acc in train_loader:
                optimizer.zero_grad()
                all_states = torch.cat([state, preceding_state[:,1].unsqueeze(1)], dim=1)
                pred_acc = self.neural_system(all_states)
                loss = criterion(pred_acc.squeeze(), acc.squeeze())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.8f}")

        torch.save(self.neural_system.state_dict(), "model_weights/neural_dynamics.pth")

        print("Neural dynamics training completed!")

class StringStabilityTrainer(pl.LightningModule):
    def __init__(self, V_net, controllers, system, current_index, learning_rate=1e-3):
        super().__init__()
        self.automatic_optimization = False
        # Save networks as module attributes so they're included in checkpoints
        self.V_net = V_net
        self.controllers = controllers  # Now it's already a ModuleList
        self.system = system
        self.learning_rate = learning_rate
        self.current_index = current_index
        
    def vector_lyapunov_conditions(self, states, x_stars, disturbances):
        """
        Verify vector Lyapunov conditions for string stability
        states: [batch_size, num_vehicles, 2]
        x_stars: [batch_size, num_vehicles, 2]
        disturbances: [batch_size, num_vehicles]
        """
        # Current Lyapunov values
        V_current = self.V_net(states, x_stars)

        # Compute control inputs
        controls = []
        for i in range(states.shape[1]):
            state_i = states[:, i, :]
            x_star_i = x_stars[:, i, :]
            controller = self.controllers[i]
            
            u_star = torch.zeros(1, device=states.device)
            u_bounds = (torch.tensor(-5.0, device=states.device), 
                       torch.tensor(5.0, device=states.device))
            
            if isinstance(controller, NetworkController):  # Check if it's a NetworkController
                control = controller(states, x_stars, u_star, u_bounds)
                controls.append(control)
            else:
                controls.append(None)
        
        # Get next states
        next_states = self.system.next_state(states, controls, disturbances)

        # Next Lyapunov values
        V_next = self.V_net(next_states, x_stars)
        
        # Compute Lyapunov decrease and larger or equal to zero conditions
        V_decreases = []
        #V_diff = torch.sum(nn.ReLU(V_current - beta))
        for i in range(1, states.shape[1]):  # Skip leading vehicle
            decrease = V_next[i-1] - V_current[i-1]
            # Add interconnection terms based on connection matrix
            decrease += 0.6 * V_current[i-1]
            for j in self.system.connections[i]:
                decrease -= self.system.connections[i][j] * V_current[j-1]
            # Add disturbance term
            #decrease -= torch.norm(disturbances[i])**2
            V_decreases.append(decrease)
            
        return torch.stack(V_decreases), V_current

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        states, x_stars, disturbances = batch

        # Compute vector Lyapunov conditions
        V_decreases, V_current = self.vector_lyapunov_conditions(states, x_stars, disturbances)

        # Compute loss ensuring string stability conditions
        loss = torch.relu(V_decreases).mean() + 0.1 * torch.relu(-V_current).mean()
        
        # Update networks
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        # Add detailed logging
        self.log("train_loss", loss, prog_bar=True)  # Show in progress bar
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        states, x_stars, disturbances = batch
        V_decreases, V_current = self.vector_lyapunov_conditions(states, x_stars, disturbances)
        val_loss = torch.relu(V_decreases + 1e-4).mean() + 0.1 * torch.relu(-V_current).mean()
        
        # Log validation loss - this is crucial for ModelCheckpoint
        self.log('val_loss', val_loss, prog_bar=True)
        
        return val_loss

    def configure_optimizers(self):
        # Collect parameters from both controllers and V_net
        parameters = []
        
        # Add V_net parameters
        parameters.extend(self.V_net.parameters())
        
        if self.current_index > 1:
            # Add controller parameters using index
            for i in range(len(self.controllers)):
                if self.controllers[i] is not None and hasattr(self.controllers[i], 'parameters'):
                    parameters.extend(self.controllers[i].parameters())

        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        return optimizer

class PlatoonDataModule(pl.LightningDataModule):
    def __init__(self, num_vehicles, cav_indices, dynamics_params, 
                 batch_size=32, num_samples=20000):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.cav_indices = cav_indices
        self.dynamics_params = dynamics_params
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        # Define state ranges
        self.spacing_range = (15.0, 25.0)  # Centered around desired_spacing
        self.vel_range = (0.0, 30.0)
        self.dist_range = (-0.5, 0.5)
        
    def _generate_samples(self):
        """Generate random samples for training"""
        states = []
        x_stars = []
        disturbances = []
        
        # Generate lead vehicle states first
        lead_spacing = torch.ones(self.num_samples)*20.0  # Reference spacing
        lead_vel = torch.ones(self.num_samples)*15.0#torch.FloatTensor(self.num_samples).uniform_(*self.vel_range)
        states.append(torch.stack([lead_spacing, lead_vel], dim=1))
        x_star = torch.zeros(2)
        x_star[0] = 20.0  # Equilibrium spacing
        x_star[1] = 15.0  # Equilibrium velocity
        x_stars.append(x_star)

        # Generate following vehicles using spacing and velocity
        for i in range(1, self.num_vehicles):
            # Generate random spacing
            spacing = torch.FloatTensor(self.num_samples).uniform_(*self.spacing_range)
            vel = torch.FloatTensor(self.num_samples).uniform_(*self.vel_range)
            
            state = torch.stack([spacing, vel], dim=1)
            states.append(state)
            
            # Generate equilibrium points
            x_star = torch.zeros(2)
            x_star[0] = 20.0  # Equilibrium spacing
            x_star[1] = 15.0  # Equilibrium velocity
            x_stars.append(x_star)
            
            # Generate disturbances
            dist = torch.zeros(self.num_samples)#torch.FloatTensor(self.num_samples).uniform_(*self.dist_range)
            disturbances.append(dist)
            
        return states, x_stars, disturbances
    
    def setup(self, stage=None):
        # Generate data
        states, x_stars, disturbances = self._generate_samples()
        train_size = int(0.8 * self.num_samples)
        # Create datasets with size as first dimension
        states = torch.stack(states)
        # Repeat x_stars for each sample
        x_stars = torch.stack(x_stars).unsqueeze(1).repeat(1, self.num_samples, 1)
        disturbances = torch.stack(disturbances)

        self.train_data = (
            states[:, :train_size,:].transpose(0, 1),  # [train_size, num_vehicles, 2]
            x_stars[:, :train_size,:].transpose(0, 1),                  # [train_size, num_vehicles, 2]
            disturbances[:, :train_size].transpose(0, 1)  # [train_size, num_vehicles]
        )
        
        self.val_data = (
            states[:, train_size:,:].transpose(0, 1),  # [val_size, num_vehicles, 2]
            x_stars[:, train_size:,:].transpose(0, 1),  # [val_size, num_vehicles, 2]
            disturbances[:, train_size:].transpose(0, 1)  # [val_size, num_vehicles]
        )

        torch.save(self.train_data, "data/train_data.pt")
        torch.save(self.val_data, "data/val_data.pt")
    
    def train_dataloader(self):
        return DataLoader(
            TensorDataset(*self.train_data),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            TensorDataset(*self.val_data),
            batch_size=self.batch_size
        )

def create_platoon_connections(num_vehicles, cav_indices):
    """
    Create connection matrix for platoon
    num_vehicles: total number of vehicles
    cav_indices: indices of CAVs in the platoon
    
    Returns: 
    Dictionary of dictionaries representing weighted connections
    {i: {j: weight}} means vehicle i is influenced by vehicle j with weight
    """
    connections = {i: {} for i in range(num_vehicles)}
    
    # Leading vehicle has no connections
    
    # Following vehicles
    for i in range(1, num_vehicles):
        if i in cav_indices:
            # CAVs can potentially connect to multiple vehicles
            connections[i][i-1] = 0#0.2  # Connection to immediate predecessor
            if i > 1:
                connections[i][i-2] = 0#0.1  # Connection to second predecessor
            if i < num_vehicles - 1:
                connections[i][i+1] = 0#0.2  # Connection to follower
        else:
            # HDVs only connect to immediate predecessor
            connections[i][i-1] = 0#0.3
            
    return connections

def train_model(num_vehicles, cav_indices, state_dims, control_dims, dynamics_params, learning_rate, batch_size, num_epochs, system_dynamics_network=None, train_system=False, index = 0, pre_trained_model = None):
    """
    Train the platoon control system using PyTorch Lightning
    
    Args:
        num_vehicles (int): Number of vehicles in platoon
        cav_indices (list): Indices of CAVs in the platoon
        state_dims (list): Dimensions of state space for each vehicle
        control_dims (list): Dimensions of control input for each vehicle
        dynamics_params (dict): Parameters for system dynamics
        learning_rate (float): Learning rate for optimization
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
    
    Returns:
        controllers (nn.ModuleList): Trained controllers
        system (PlatoonDynamics): Initialized system dynamics
    """
    # Create connection matrix
    connection_matrix = create_platoon_connections(num_vehicles, cav_indices)

    # Initialize networks
    V_net = VectorLyapunovNetwork(state_dims)
    controllers = nn.ModuleList([
        NetworkController(sum(state_dims), control_dims[i]) if i in cav_indices 
        else nn.Identity() for i in range(num_vehicles)
    ])
    if pre_trained_model is not None:
        controllers[1].load_state_dict(pre_trained_model)

    # Initialize system dynamics
    if system_dynamics_network is not None:
        system = PlatoonDynamics(dynamics_params, connection_matrix, if_neural_network=True, neural_system=system_dynamics_network, train_system=train_system)
    else:
        system = PlatoonDynamics(dynamics_params, connection_matrix)

    # Initialize data module
    data_module = PlatoonDataModule(num_vehicles, cav_indices, dynamics_params, 
                                  batch_size=batch_size)

    # Initialize trainer
    trainer = StringStabilityTrainer(V_net, controllers, system, 
                                   learning_rate=learning_rate, current_index = index)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_weights',
        filename='best_model',
        save_top_k=1,
        mode='min'
    )

    # Train the system
    pl_trainer = pl.Trainer(
        max_epochs=num_epochs,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
    )
    pl_trainer.fit(trainer, data_module)

    # load new controllers and V_net
    controllers = trainer.controllers
    V_net = trainer.V_net

    return controllers, system, V_net

class PlatoonDataModuleRetrain(pl.LightningDataModule):
    def __init__(self, epoch, counterexamples, counterexample_ranges, batch_size=32, 
                 num_points=50000):
        super().__init__()
        self.num_points = num_points
        self.counterexamples = counterexamples
        self.counterexample_ranges = counterexample_ranges
        self.in_train_file = "data/train_data.pt"
        self.out_train_file = "data/train_data.pt"
        self.out_val_file = "data/val_data.pt"
        self.epoch = epoch
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 加载原有训练数据
        old_data_states = torch.load(self.in_train_file)[0]
        old_data_x_stars = torch.load(self.in_train_file)[1]
        old_data_disturbances = torch.load(self.in_train_file)[2]

        # generate new x_stars for counterexamples, which is of same size as counterexamples
        new_x_stars = torch.zeros_like(self.counterexamples)
        new_x_stars[..., 0] = 15.0
        new_x_stars[..., 1] = 20.0

        new_data_disturbances = torch.zeros_like(self.counterexamples[:,0,:].squeeze())

        # 添加反例数据
        combined_data_states = torch.cat([old_data_states, self.counterexamples], dim=0)
        combined_data_x_stars = torch.cat([old_data_x_stars, new_x_stars], dim=0)
        combined_data_disturbances = torch.cat([old_data_disturbances, new_data_disturbances], dim=0)

        combined_data = (combined_data_states, combined_data_x_stars, combined_data_disturbances)
        
        # 保存新的训练数据
        torch.save(combined_data, self.out_train_file)
        
        # 创建数据集
        self.train_dataset = TensorDataset(*combined_data)
        self.val_dataset = TensorDataset(*torch.load(self.out_val_file))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

def retrain_model(num_vehicles, cav_indices, state_dims, control_dims, dynamics_params,
                 counterexamples, counterexample_ranges, epoch,
                 in_model, in_controller,
                 learning_rate=1e-4, batch_size=32, system_dynamics_network=None, index = 0):
    """
    Retrain the platoon control system using counterexamples
    
    Args:
        num_vehicles (int): Number of vehicles in platoon
        cav_indices (list): Indices of CAVs in the platoon
        state_dims (list): Dimensions of state space for each vehicle
        control_dims (list): Dimensions of control input for each vehicle
        dynamics_params (dict): Parameters for system dynamics
        counterexamples (tensor): Counterexamples found during verification
        counterexample_ranges (tensor): Ranges for the counterexamples
        epoch (int): Current training epoch
        in_model: Input Lyapunov network (V_net)
        in_controller: Input controller network
        learning_rate (float): Learning rate for optimization
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
    
    Returns:
        controllers (nn.ModuleList): Retrained controllers
        system (PlatoonDynamics): System dynamics
        V_net (VectorLyapunovNetwork): Retrained Lyapunov network
    """
    # Create connection matrix
    connection_matrix = create_platoon_connections(num_vehicles, cav_indices)

    # Use provided networks directly
    V_net = in_model
    controllers = in_controller

    # Initialize system dynamics
    if system_dynamics_network is not None:
        system = PlatoonDynamics(dynamics_params, connection_matrix, if_neural_network=True, neural_system=system_dynamics_network)
    else:
        system = PlatoonDynamics(dynamics_params, connection_matrix)

    # Initialize data module with counterexamples
    data_module = PlatoonDataModuleRetrain(
        epoch, counterexamples, counterexample_ranges,
        batch_size=batch_size
    )

    # Initialize trainer for retraining
    trainer = StringStabilityTrainer(V_net, controllers, system, 
                                learning_rate=learning_rate, current_index = index)
    #trainer = StringStabilityTrainerRetrain(
    #    V_net, controllers,
    #    primal_learning_rate=learning_rate
    #)

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_weights',
        filename=f'best_model',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    # Train the system
    pl_trainer = pl.Trainer(
        max_epochs=epoch,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
    )
    
    pl_trainer.fit(trainer, data_module)
    controllers = trainer.controllers
    V_net = trainer.V_net
    
    return controllers, system, V_net


if __name__ == "__main__":
    # System parameters
    num_vehicles = 3
    cav_indices = [1]  # Second vehicle is CAV
    state_dims = [2] * num_vehicles  # Each vehicle has 2 states (position, velocity)
    control_dims = [1] * num_vehicles  # Each vehicle has 1 control input (acceleration)

    # Dynamics parameters
    dynamics_params = {
        'dt': 0.1,
        'alpha': 0.6,
        'beta': 0.9,
        'v_max': 30.0,
        's_st': 5.0,
        's_go': 35.0,
        'a_max': 5.0,
        'a_min': -5.0,
        'desired_spacing': 20.0
    }

    # Training parameters
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 100

    # Train the model
    controllers, system, V_net = train_model(
        num_vehicles=num_vehicles,
        cav_indices=cav_indices,
        state_dims=state_dims,
        control_dims=control_dims,
        dynamics_params=dynamics_params,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs
    )


    print("Training completed and model saved!")

    

        



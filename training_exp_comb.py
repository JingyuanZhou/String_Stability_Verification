import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from networks import NetworkController, VectorLyapunovNetwork, system_network, DoubleQCritic
import torch.onnx

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
        self.a_max = dynamics_params.get('a_max', 100.0)
        self.a_min = dynamics_params.get('a_min', -100.0)
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
    
    def next_state(self, states, controls, disturbances, eval = False):
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
                #print("true_acc_i", acc_i)
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
    def __init__(self, V_net, controllers, system, current_index, learning_rate=1e-3, original_controller = None, critics = None):
        super().__init__()
        self.automatic_optimization = False
        # Save networks as module attributes so they're included in checkpoints
        self.V_net = V_net
        self.controllers = controllers  # Now it's already a ModuleList
        self.system = system
        self.learning_rate = learning_rate
        self.current_index = current_index
        self.original_controller = original_controller
        self.critics = critics
        self.dt = 0.1
        
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
    
    def vector_lyapunov_conditions(self, states, x_stars, disturbances):
        """
        Verify vector Lyapunov conditions for string stability
        
        states: [batch_size, num_vehicles, 2]
        x_stars: [batch_size, num_vehicles, 2]
        disturbances: [batch_size, num_vehicles]
        """
        if_fixed_coupling = False
        if if_fixed_coupling:
            coupling_matrix = self.system.connections
        else:
            G = self.create_binary_adjacency_matrix(self.system.connections)
            coupling_matrix = self.V_net.coupling_matrix(G)
        # Current Lyapunov values
        V_current = self.V_net(states, x_stars)

        # Compute control inputs
        controls = []
        original_controls = []
        for i in range(states.shape[1]):
            state_i = states[:, i, :]
            x_star_i = x_stars[:, i, :]
            controller = self.controllers[i]
            original_controller = self.original_controller[i]
            
            u_star = torch.zeros(1, device=states.device)
            u_bounds = (torch.tensor(-5.0, device=states.device), 
                       torch.tensor(5.0, device=states.device))
            
            if isinstance(controller, NetworkController):  # Check if it's a NetworkController
                control = controller(states, x_stars, u_star, u_bounds)
                controls.append(control)
                original_control = original_controller(states, x_stars, u_star, u_bounds)
                original_controls.append(original_control)
            else:
                controls.append(None)
                original_controls.append(None)
        
        control_dist = torch.square(original_controls[1] - controls[1]).mean()/2000
        value_dist = 0*torch.relu(-(self.critics(states, controls[1]) - self.critics(states, original_controls[1]))).mean()

        # Get next states
        next_states = self.system.next_state(states, controls, disturbances)

        # Next Lyapunov values
        V_next = self.V_net(next_states, x_stars)
        
        # Compute Lyapunov decrease and larger or equal to zero conditions
        V_decreases = []
        coef_cons = []
        #V_diff = torch.sum(nn.ReLU(V_current - beta))

        for i in range(1, states.shape[1]):  # Skip leading vehicle
            decrease = (V_next[:,i-1] - V_current[:,i-1])
            # Add interconnection terms based on connection matrix
            decrease += 0.05 * V_current[:,i-1]
            coef_con = torch.tensor(-0.05, device=V_current.device, dtype=V_current.dtype)
            if if_fixed_coupling:
                for j in coupling_matrix[i]:
                    if j >= 1:
                        decrease -= coupling_matrix[i][j] * V_current[:,j-1]
            else:
                for j in range(1, states.shape[1]):
                    decrease -= coupling_matrix[i][j] * V_current[:,j-1]
                    coef_con += coupling_matrix[i][j]
            # Add disturbance term
            #decrease -= torch.norm(disturbances[i])**2
            V_decreases.append(decrease)
            coef_cons.append(coef_con)
            
        return torch.stack(V_decreases), V_current, control_dist, torch.stack(coef_cons), value_dist

    def cal_reward_objective(self, states, x_stars, disturbances):
        """
        Calculate reward objective for other objectives
        """

        spacing = states[:, :, 0]
        velocity = states[:, :, 1]
        cav_index = 1

        '''
        ttc = spacing[:,cav_index] / (velocity[:,cav_index-1] - velocity[:,cav_index] + 1e-6)
        safety_list = []

        for i in range(0, states.shape[0]):
            if 0 < ttc[i] < 4:
                safety = torch.log(ttc[i] / 4)
            else:
                safety = 0
            safety_list.append(safety)

        efficiency_list = []
        for i in range(0, states.shape[1]):
            efficiency = 0
            if self.spacing[i,1]/self.velocity[i,1] > 2.5:  # 车距过大惩罚
                efficiency -= 2.5
            efficiency_list.append(efficiency)
        '''

        #stability = 0
        # calculate a decay weights for stability
        #decay_weights = np.linspace(0.6, 0.1, 3 - cav_index)
        #for i in range(cav_index, cav_index+2):
        #    stability = decay_weights[i - cav_index] * (velocity[:,i] - velocity[:,i-1])**2

        reward_weights = [0.3, 0.3, 0.1]

        reward = 0#reward_weights[2] * stability.mean()
        
        return reward

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        states, x_stars, disturbances = batch

        # Compute vector Lyapunov conditions
        epsilon = 1e-2
        V_decreases, V_current, control_dist, coef_cons, value_dist = self.vector_lyapunov_conditions(states, x_stars, disturbances)

        reward_related_loss = self.cal_reward_objective(states, x_stars, disturbances)

        # Compute loss ensuring string stability conditions
        if self.current_index == 0:
            loss = 10 * torch.relu(-V_current+epsilon).mean() + 5*torch.relu(V_decreases + epsilon).mean()# + 100*torch.relu(coef_cons+epsilon).mean() + control_dist #+ value_dist 
        else: 
            loss = 10 * torch.relu(-V_current+epsilon).mean() + 5*torch.relu(V_decreases + epsilon).mean() + control_dist #+ 100*torch.relu(coef_cons+epsilon).mean()#+ value_dist + reward_related_loss 
        
        # Update networks
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        # Add detailed logging
        self.log("train_loss", loss, prog_bar=True)  # Show in progress bar
        self.log("loss_decrease", 5*torch.relu(V_decreases).mean(), prog_bar=True)
        self.log("loss_positive", 10*torch.relu(-V_current).mean(), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        states, x_stars, disturbances = batch
        epsilon = 1e-2
        reward_related_loss = self.cal_reward_objective(states, x_stars, disturbances)
        V_decreases, V_current, control_dist, coef_cons, value_dist = self.vector_lyapunov_conditions(states, x_stars, disturbances)
        if self.current_index == 0:
            val_loss = 10 * torch.relu(-V_current+epsilon).mean() +  5*torch.relu(V_decreases + epsilon).mean() # +control_dist+ value_dist + torch.relu(coef_cons).mean()
        else: 
            val_loss = 10 * torch.relu(-V_current+epsilon).mean() + 5*torch.relu(V_decreases + epsilon).mean() #+ control_dist+ value_dist + reward_related_loss + torch.relu(coef_cons).mean()    
        
        # Log validation loss - this is crucial for ModelCheckpoint
        self.log('val_loss', val_loss, prog_bar=True)
        
        return val_loss

    def configure_optimizers(self):
        # Collect parameters from both controllers and V_net
        parameters = []
        
        # Add V_net parameters
        parameters.extend(self.V_net.parameters())
        
        if self.current_index > 0:
            # Add controller parameters using index
            for i in range(len(self.controllers)):
                if self.controllers[i] is not None and hasattr(self.controllers[i], 'parameters'):
                    parameters.extend(self.controllers[i].parameters())

        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        return optimizer

class PlatoonDataModule(pl.LightningDataModule):
    def __init__(self, num_vehicles, cav_indices, dynamics_params, 
                 batch_size=16, num_samples=2000):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.cav_indices = cav_indices
        self.dynamics_params = dynamics_params
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        # Define state ranges
        self.spacing_range = (5, 35.0)  # Centered around desired_spacing
        self.vel_range = (5.0, 25.0)
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
            connections[i][i-1] = 0.01#0.2  # Connection to immediate predecessor
            if i > 1:
                connections[i][i-2] = 0.01#0.1  # Connection to second predecessor
            if i < num_vehicles - 1:
                connections[i][i+1] = 0.02#0.2  # Connection to follower
        else:
            # HDVs only connect to immediate predecessor
            connections[i][i-1] = 0.02#0.3
            
    return connections

def train_model(num_vehicles, cav_indices, state_dims, control_dims, dynamics_params, learning_rate, batch_size, num_epochs, system_dynamics_network=None, train_system=False, index = 0, pre_trained_model = None, pre_trained_critics = None):
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
        raw_parameters = torch.load(pre_trained_model)
        controller_parameters = {}
        for k, v in raw_parameters.items():
            if k.startswith('trunk'):
                new_key = k.replace('trunk', '1.network')
                # 如果是最后一层的参数，只取一半（对应均值输出）

                if '1.network.4.weight' in new_key:  
                    controller_parameters[new_key] = v[:1, :]  # 只保留第一行，对应均值
                elif '1.network.4.bias' in new_key:
                    controller_parameters[new_key] = v[:1]  # 只保留第一个元素，对应均值
                else:
                    controller_parameters[new_key] = v
        controllers.load_state_dict(controller_parameters)

    critics = DoubleQCritic(sum(state_dims), control_dims[1])
    if pre_trained_critics is not None:
        raw_parameters_critics = torch.load(pre_trained_critics)
        critics.load_state_dict(raw_parameters_critics)

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
                                   learning_rate=learning_rate, current_index = index, original_controller = controllers, critics = critics)

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

    # Update system connections
    # Convert current dictionary connections to a binary adjacency matrix
    G = torch.zeros(len(system.connections), len(system.connections))
    for i in system.connections:
        for j in system.connections[i]:
            G[i, j] = 1.0
    
    # Compute new coupling matrix using V_net
    new_matrix = V_net.coupling_matrix(G)
    
    # Convert new_matrix back to dictionary form
    new_connections = {}
    for i in range(new_matrix.size(0)):
        new_connections[i] = {}
        for j in range(new_matrix.size(1)):
            val = new_matrix[i, j].item()
            if abs(val) > 1e-9:
                new_connections[i][j] = val

    print("Old connections: ", system.connections)
    print("New connections: ", new_connections)
    system.connections = new_connections

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
        new_x_stars[..., 0] = 20.0
        new_x_stars[..., 1] = 15.0

        new_data_disturbances = torch.zeros_like(self.counterexamples[:,0,:].squeeze())

        # 添加反例数据
        combined_data_states = torch.cat([old_data_states, self.counterexamples], dim=0)
        combined_data_x_stars = torch.cat([old_data_x_stars, new_x_stars], dim=0)
        combined_data_disturbances = torch.cat([old_data_disturbances, new_data_disturbances], dim=0)

        # add some data augementation for original counterexamples
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

def check_counter_examples(V_net, controllers, system, cav_indices, counterexamples,combined_model_path):
    # check if the counterexamples are really counterexamples
    number_of_false_counterexamples = 0
    
    combined_model = torch.load(combined_model_path.replace(".onnx", ".pth"))

    for i in range(counterexamples.shape[0]):

        state = counterexamples[i]
        x_star = torch.zeros_like(state)
        x_star[...,0] = 20.0
        x_star[...,1] = 15.0
        disturbances = torch.zeros_like(state[0])
        V_values = V_net(state, x_star)[0]
        controls = []
        for j in range(state.shape[0]):
            if j in cav_indices:
                controller = controllers[j]
                u_star = torch.zeros(1)
                u_bounds = (torch.tensor(-5.0), torch.tensor(5.0))
                control = controller(state, x_star, u_star, u_bounds)
                controls.append(control)
            else:
                controls.append(None)
        
        next_states = system.next_state(state.unsqueeze(0), controls, disturbances.unsqueeze(0))
        cmb_output_V, cmb_next_state, cmb_next_V = combined_model(state)
        #print("true_current_state:",state," true_next_state: ", next_states, "cmb_next_state: ", cmb_next_state)
        #print("true_current_V:", V_values, " true_next_V: ", V_net(next_states, x_star)[0], "cmb_next_V: ", cmb_next_V)
        V_next = V_net(next_states, x_star)[0]
        false_ce = True
        for i in range(V_next.size(0)):
            aii = 0.05
            epsilon = 0.0
            vars_ = [V_next[i].item(), V_values[i].item()]
            coeffs = [1.0, -1.0 + aii]

            for j in system.connections[i+1]:
                if j >= 1:
                    vars_.append(V_values[j-1].item())
                    coeffs.append(-system.connections[i+1][j])

            expr = sum(v * c for v, c in zip(vars_, coeffs))
            if expr >= epsilon or V_values[i] <= 0.0 or V_next[i] <= 0.0:
                false_ce = False
                #print(f"vars: {[round(v, 3) for v in vars_]}, coeffs: {[round(c, 3) for c in coeffs]}, expr: {round(expr, 3)}")
        if false_ce:
            number_of_false_counterexamples += 1

    if number_of_false_counterexamples == 0:
        print("All counterexamples are valid!")
    else:
        print(f"{number_of_false_counterexamples} false counterexamples found!")
        
def add_noise_to_counterexamples(counterexamples):
    """
    Add noise to counterexamples for data augmentation
    
    Args:
        counterexamples (tensor): Counterexamples to add noise
        expansion_factor (float): Maximum noise level as a fraction of the counterexample range
    
    Returns:
        counterexamples (tensor): Augmented counterexamples
    """
    
    counter_example_expanded = counterexamples
    for i in range(5):
        noise = (torch.rand_like(counterexamples) - 0.5) * 0.01 * (i+1)
        noise[:, 0, 0] = 0.0  # No noise on first vehicle spacing
        noise[:, 0, 1] = 0.0  # No noise on first vehicle velocity
        counter_example_expanded = torch.cat([counter_example_expanded, counterexamples + noise], dim=0)
    print("original counterexamples: ", counterexamples.shape)
    print("expanded counterexamples: ", counter_example_expanded.shape)
    return counter_example_expanded

def retrain_model(num_vehicles, cav_indices, state_dims, control_dims, in_system,
                 counterexamples, counterexample_ranges, epoch,
                 in_model, in_controller,
                 learning_rate=1e-4, batch_size=32, index = 0, pre_trained_model = None, pre_trained_critics = None, combined_model_path = None):
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
    #connection_matrix = create_platoon_connections(num_vehicles, cav_indices)

    # Use provided networks directly
    V_net = in_model
    controllers = in_controller

    # Initialize system dynamics
    system = in_system

    #check_counter_examples(V_net, controllers, system, cav_indices, counterexamples, combined_model_path)
    counterexamples = add_noise_to_counterexamples(counterexamples)
    # Initialize data module with counterexamples
    data_module = PlatoonDataModuleRetrain(
        epoch, counterexamples, counterexample_ranges,
        batch_size=batch_size
    )

    original_controllers = nn.ModuleList([
        NetworkController(sum(state_dims), control_dims[i]) if i in cav_indices 
        else nn.Identity() for i in range(num_vehicles)
    ])

    if pre_trained_model is not None:
        raw_parameters = torch.load(pre_trained_model)
        controller_parameters = {}
        for k, v in raw_parameters.items():
            if k.startswith('trunk'):
                new_key = k.replace('trunk', '1.network')
                # 如果是最后一层的参数，只取一半（对应均值输出）

                if '1.network.4.weight' in new_key:  
                    controller_parameters[new_key] = v[:1, :]  # 只保留第一行，对应均值
                elif '1.network.4.bias' in new_key:
                    controller_parameters[new_key] = v[:1]  # 只保留第一个元素，对应均值
                else:
                    controller_parameters[new_key] = v
        original_controllers.load_state_dict(controller_parameters)

    critics = DoubleQCritic(sum(state_dims), control_dims[1])
    if pre_trained_critics is not None:
        raw_parameters_critics = torch.load(pre_trained_critics)
        critics.load_state_dict(raw_parameters_critics)

    # Initialize trainer for retraining
    trainer = StringStabilityTrainer(V_net, controllers, system, 
                                learning_rate=learning_rate, current_index = index, original_controller = original_controllers, critics = critics)
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
        save_last=False
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

        # Update system connections
    # Convert current dictionary connections to a binary adjacency matrix
    G = torch.zeros(len(system.connections), len(system.connections))
    for i in system.connections:
        for j in system.connections[i]:
            G[i, j] = 1.0
    
    # Compute new coupling matrix using V_net
    new_matrix = V_net.coupling_matrix(G)
    
    # Convert new_matrix back to dictionary form
    new_connections = {}
    for i in range(new_matrix.size(0)):
        new_connections[i] = {}
        for j in range(new_matrix.size(1)):
            val = new_matrix[i, j].item()
            if abs(val) > 1e-9:
                new_connections[i][j] = val

    print("Old connections: ", system.connections)
    print("New connections: ", new_connections)
    system.connections = new_connections

    
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

    

        



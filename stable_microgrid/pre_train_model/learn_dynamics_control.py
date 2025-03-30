from microgrid_simulate import MicrogridSimulator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Create output directories if they don't exist
os.makedirs('output_figures', exist_ok=True)

# ==================== Neural Network Models ====================
class DynamicsNN(nn.Module):
    """Neural network for learning microgrid dynamics"""
    def __init__(self, state_dim=3, neighbor_dim=6, control_dim=1, hidden_dim=64):
        super(DynamicsNN, self).__init__()
        self.input_dim = state_dim + neighbor_dim + control_dim  # [xi, xNi, ui]
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, neighbor_state, control):
        # state: [batch, 3] (δ, ω, ξ)
        # neighbor_state: [batch, 2, 3] (2 neighbors, each with δ, ω, ξ)
        # control: [batch, 1]
        
        # Flatten neighbor states
        neighbor_flat = neighbor_state.reshape(neighbor_state.shape[0], -1)
        
        # Concatenate all inputs
        x = torch.cat([state, neighbor_flat, control.unsqueeze(-1)], dim=-1)
        
        # Predict state derivatives
        return self.net(x)

class ControllerNN(nn.Module):
    """Neural network for learning UAV controller"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ControllerNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, error_state):
        # Concatenate position and velocity errors
        return self.net(error_state)
        

# ==================== Data Collection ====================
class DataCollector:
    def __init__(self, simulator: MicrogridSimulator):
        """
        Initialize the data collector.
        
        Args:
            simulator: MicrogridSimulator instance
        """
        self.simulator = simulator
        self.n = simulator.n  # Number of inverters
        
    def collect_monte_carlo_data(self, n_trajectories: int = 3000, 
                                dt: float = 0.01, range_noise: float = 5) -> Dict[str, np.ndarray]:
        """
        Collect state and control data using Monte Carlo sampling.
        One sample per trajectory.
        
        Args:
            n_trajectories: Number of trajectories/samples to collect
            dt: Time step size
            range_noise: Range of the noise
            
        Returns:
            Dictionary containing:
            - current_states: Array of shape (n_trajectories, n_inverters, 3)
            - next_states: Array of shape (n_trajectories, n_inverters, 3)
            - current_neighbor_states: Array of shape (n_trajectories, n_inverters, 2, 3)
            - next_neighbor_states: Array of shape (n_trajectories, n_inverters, 2, 3)
            - controls: Array of shape (n_trajectories, n_inverters)
        """
        # Initialize data arrays
        current_states = np.zeros((n_trajectories, self.n, 3))  # [δ, ω, ξ]
        next_states = np.zeros((n_trajectories, self.n, 3))
        current_neighbor_states = np.zeros((n_trajectories, self.n, 2, 3))
        next_neighbor_states = np.zeros((n_trajectories, self.n, 2, 3))
        controls = np.zeros((n_trajectories, self.n))
        
        for traj in tqdm(range(n_trajectories), desc="Collecting samples"):
            # Reset simulator state to equilibrium
            self.simulator.delta = np.zeros(self.n)  # δ = 0
            self.simulator.omega = np.array([self.simulator.omega_star] * self.n)  # ω = ω*
            self.simulator.xi = np.zeros(self.n)  # ξ = 0
            
            # Generate phase angles sequentially
            # First node stays at 0
            self.simulator.delta[0] = 0.0
            
            # Second node: random in [0, π/2]
            self.simulator.delta[1] = np.random.uniform(0, np.pi/2)
            
            # Remaining nodes: random in [prev_node, prev_node + π/2]
            for i in range(2, self.n):
                self.simulator.delta[i] = np.random.uniform(
                    self.simulator.delta[i-1],
                    self.simulator.delta[i-1] + np.pi/2
                )
            
            # Add perturbations to other states
            for i in range(self.n):
                self.simulator.omega[i] = self.simulator.omega_star + np.random.uniform(-range_noise*20, range_noise*20)
                self.simulator.xi[i] = np.random.uniform(-range_noise, range_noise)
            
            # Store current state
            current_states[traj, :, 0] = self.simulator.delta
            current_states[traj, :, 1] = self.simulator.omega
            current_states[traj, :, 2] = self.simulator.xi
            
            # Store neighbor states
            for i in range(self.n):
                if i > 0:  # Has left neighbor
                    current_neighbor_states[traj, i, 0] = current_states[traj, i-1]
                if i < self.n-1:  # Has right neighbor
                    current_neighbor_states[traj, i, 1] = current_states[traj, i+1]
            
            # Get control input
            controls[traj] = self.simulator.secondary_droop_control(
                self.simulator.omega, self.simulator.xi, self.simulator.delta
            )
            
            # Step simulation
            delta_next, omega_next, xi_next = self.simulator.step_without_update_loads(dt, 0)
            
            # Store next state
            next_states[traj, :, 0] = delta_next
            next_states[traj, :, 1] = omega_next
            next_states[traj, :, 2] = xi_next
            
            # Store next neighbor states
            for i in range(self.n):
                if i > 0:  # Has left neighbor
                    next_neighbor_states[traj, i, 0] = next_states[traj, i-1]
                if i < self.n-1:  # Has right neighbor
                    next_neighbor_states[traj, i, 1] = next_states[traj, i+1]
        
        return {
            'current_states': current_states,
            'next_states': next_states,
            'current_neighbor_states': current_neighbor_states,
            'next_neighbor_states': next_neighbor_states,
            'controls': controls
        }

class MicrogridDataset(Dataset):
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        """
        Create a PyTorch dataset from collected data.
        
        Args:
            data_dict: Dictionary containing simulation data
        """
        self.current_states = torch.FloatTensor(data_dict['current_states'])
        self.next_states = torch.FloatTensor(data_dict['next_states'])
        self.current_neighbor_states = torch.FloatTensor(data_dict['current_neighbor_states'])
        self.next_neighbor_states = torch.FloatTensor(data_dict['next_neighbor_states'])
        self.controls = torch.FloatTensor(data_dict['controls'])
        
    def __len__(self):
        return len(self.current_states)
    
    def __getitem__(self, idx):
        return {
            'current_states': self.current_states[idx],
            'next_states': self.next_states[idx],
            'current_neighbor_states': self.current_neighbor_states[idx],
            'next_neighbor_states': self.next_neighbor_states[idx],
            'controls': self.controls[idx]
        }

def NN_dynamics_training(dataloader: DataLoader, 
                        num_epochs: int = 100,
                        learning_rate: float = 1e-3,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[List[DynamicsNN], List[List[float]]]:
    """
    Train neural networks to learn microgrid dynamics.
    One network for each inverter.
    
    Args:
        dataloader: DataLoader containing training data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        List of trained models and list of training losses for each model
    """
    # Initialize models, optimizers, and loss histories for each inverter
    models = []
    optimizers = []
    schedulers = []
    all_losses = []
    best_losses = []
    
    n_inverters = next(iter(dataloader))['current_states'].shape[1]
    
    for i in range(n_inverters):
        if i == 0:
            model = DynamicsNN(state_dim=3, neighbor_dim=3, control_dim=1).to(device)
        elif i == n_inverters - 1:
            model = DynamicsNN(state_dim=3, neighbor_dim=3, control_dim=1).to(device)
        else:
            model = DynamicsNN(state_dim=3, neighbor_dim=6, control_dim=1).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        all_losses.append([])
        best_losses.append(float('inf'))
    
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining {n_inverters} dynamics models on {device}")
    print("=" * 50)
    
    # Create progress bar
    pbar = tqdm(range(num_epochs), desc="Training")
    
    for epoch in pbar:
        epoch_losses = [0.0] * n_inverters
        num_batches = 0
        
        for batch in dataloader:
            # Get data
            current_states = batch['current_states'].to(device)  # [batch, n_inverters, 3]
            current_neighbor_states = batch['current_neighbor_states'].to(device)  # [batch, n_inverters, 2, 3]
            controls = batch['controls'].to(device)  # [batch, n_inverters]
            next_states = batch['next_states'].to(device)  # [batch, n_inverters, 3]
            
            # Train each inverter's model
            for i in range(n_inverters):
                # Get data for current inverter
                current_state = current_states[:, i, :]  # [batch, 3]
                if i == 0:
                    current_neighbor_state = current_neighbor_states[:, i, 1, :]  # [batch, 3]
                elif i == n_inverters - 1:
                    current_neighbor_state = current_neighbor_states[:, i, 0, :]  # [batch, 3]
                else:
                    current_neighbor_state = current_neighbor_states[:, i, :, :]  # [batch, 2, 3]
                control = controls[:, i]  # [batch]
                target = next_states[:, i, :]
                
                # Forward pass
                pred = models[i](current_state, current_neighbor_state, control)
                loss = criterion(pred, target)
                
                # Backward pass
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                
                epoch_losses[i] += loss.item()
            
            num_batches += 1
        
        # Average losses for this epoch
        avg_losses = [loss/num_batches for loss in epoch_losses]
        for i in range(n_inverters):
            all_losses[i].append(avg_losses[i])
            
            # Update learning rate scheduler
            schedulers[i].step(avg_losses[i])
            
            # Update best loss
            if avg_losses[i] < best_losses[i]:
                best_losses[i] = avg_losses[i]
        
        # Update progress bar with current losses and learning rates
        postfix_dict = {}
        for i in range(n_inverters):
            postfix_dict[f'loss_{i}'] = f'{avg_losses[i]:.6f}'
            postfix_dict[f'lr_{i}'] = f'{optimizers[i].param_groups[0]["lr"]:.2e}'
        pbar.set_postfix(postfix_dict)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    for i in range(n_inverters):
        plt.plot(all_losses[i], label=f'Inverter {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Dynamics Models Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_figures/dynamics_training_loss.png')
    plt.close()
    
    print("\nTraining completed!")
    for i in range(n_inverters):
        print(f"Inverter {i+1}:")
        print(f"  Final loss: {all_losses[i][-1]:.6f}")
        print(f"  Best loss: {min(all_losses[i]):.6f}")
        print(f"  Final learning rate: {optimizers[i].param_groups[0]['lr']:.2e}")
    
    return models, all_losses

def NN_control_training(dataloader: DataLoader,
                       num_epochs: int = 100,
                       learning_rate: float = 1e-3,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[List[ControllerNN], List[List[float]]]:
    """
    Train neural networks to learn microgrid control.
    One network for each inverter.
    
    Args:
        dataloader: DataLoader containing training data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        List of trained models and list of training losses for each model
    """
    # Initialize models, optimizers, and loss histories for each inverter
    models = []
    optimizers = []
    schedulers = []
    all_losses = []
    best_losses = []
    
    n_inverters = next(iter(dataloader))['current_states'].shape[1]
    
    for i in range(n_inverters):
        if i == 0:
            model = ControllerNN(input_dim=9, output_dim=1).to(device)  # [state(3) + neighbor(3) + target(3)]
        elif i == n_inverters - 1:
            model = ControllerNN(input_dim=9, output_dim=1).to(device)  # [state(3) + neighbor(3) + target(3)]
        else:
            model = ControllerNN(input_dim=12, output_dim=1).to(device)  # [state(3) + neighbors(6) + target(3)]
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        all_losses.append([])
        best_losses.append(float('inf'))
    
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining {n_inverters} control models on {device}")
    print("=" * 50)
    
    # Create progress bar
    pbar = tqdm(range(num_epochs), desc="Training")
    
    for epoch in pbar:
        epoch_losses = [0.0] * n_inverters
        num_batches = 0
        
        for batch in dataloader:
            # Get data
            current_states = batch['current_states'].to(device)  # [batch, n_inverters, 3]
            current_neighbor_states = batch['current_neighbor_states'].to(device)  # [batch, n_inverters, 2, 3]
            controls = batch['controls'].to(device)  # [batch, n_inverters]
            
            # Target states (equilibrium)
            target_states = torch.zeros_like(current_states)
            target_states[:, :, 1] = 2 * np.pi * 50  # ω = ω*
            
            # Train each inverter's model
            for i in range(n_inverters):
                # Get data for current inverter
                current_state = current_states[:, i, :]  # [batch, 3]
                target_state = target_states[:, i, :]  # [batch, 3]
                
                if i == 0:
                    current_neighbor_state = current_neighbor_states[:, i, 1, :]  # [batch, 3]
                    input_state = torch.cat([current_state, current_neighbor_state, target_state], dim=-1)
                elif i == n_inverters - 1:
                    current_neighbor_state = current_neighbor_states[:, i, 0, :]  # [batch, 3]
                    input_state = torch.cat([current_state, current_neighbor_state, target_state], dim=-1)
                else:
                    current_neighbor_state = current_neighbor_states[:, i, :, :].reshape(-1, 6)  # [batch, 6]
                    input_state = torch.cat([current_state, current_neighbor_state, target_state], dim=-1)
                
                # Forward pass
                pred = models[i](input_state)
                target = controls[:, i].unsqueeze(-1)  # [batch, 1]
                loss = criterion(pred, target)
                
                # Backward pass
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                
                epoch_losses[i] += loss.item()
            
            num_batches += 1
        
        # Average losses for this epoch
        avg_losses = [loss/num_batches for loss in epoch_losses]
        for i in range(n_inverters):
            all_losses[i].append(avg_losses[i])
            
            # Update learning rate scheduler
            schedulers[i].step(avg_losses[i])
            
            # Update best loss
            if avg_losses[i] < best_losses[i]:
                best_losses[i] = avg_losses[i]
        
        # Update progress bar with current losses and learning rates
        postfix_dict = {}
        for i in range(n_inverters):
            postfix_dict[f'loss_{i}'] = f'{avg_losses[i]:.6f}'
            postfix_dict[f'lr_{i}'] = f'{optimizers[i].param_groups[0]["lr"]:.2e}'
        pbar.set_postfix(postfix_dict)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    for i in range(n_inverters):
        plt.plot(all_losses[i], label=f'Inverter {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Control Models Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_figures/control_training_loss.png')
    plt.close()
    
    print("\nTraining completed!")
    for i in range(n_inverters):
        print(f"Inverter {i+1}:")
        print(f"  Final loss: {all_losses[i][-1]:.6f}")
        print(f"  Best loss: {min(all_losses[i]):.6f}")
        print(f"  Final learning rate: {optimizers[i].param_groups[0]['lr']:.2e}")
    
    return models, all_losses

if __name__ == "__main__":
    # Create simulator with default parameters
    from microgrid_simulate import MicrogridSimulator
    
    # System parameters
    n = 3  # Number of inverters
    V = [325.3] * n  # Voltage magnitude (V)
    B = [[-0.0056, -0.0112, 0],
         [-0.0112, -0.0151, -0.0039],
         [0, -0.0039, -0.0112]]  # Line susceptances
    P_L = [1260.0] * n  # Initial loads (W)
    P_star = [1260.0] * n  # Desired power injections (W)
    omega_star = 2 * np.pi * 50  # Desired frequency (rad/s) - 50 Hz
    
    # Controller parameters
    tau = [1.4895] * n  # Time constants
    eta = [6.3509e-4] * n  # Droop gains
    k = [4.9481] * n  # Secondary control gains
    
    # Communication graph parameters
    g_val = 0.0213
    l_val = 0.0043
    
    # Construct communication graphs
    G = np.zeros((n, n))
    L = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            G[i,i-1] = G[i-1,i] = -g_val
            L[i,i-1] = L[i-1,i] = -l_val
        if i < n-1:
            G[i,i+1] = G[i+1,i] = -g_val
            L[i,i+1] = L[i+1,i] = -l_val
        G[i,i] = -np.sum(G[i,:])
        L[i,i] = -np.sum(L[i,:])
    
    # Create simulator
    simulator = MicrogridSimulator(n, V, B, P_L, P_star, omega_star, tau, eta, k, G, L)
    
    # Create data collector
    collector = DataCollector(simulator)
    
    # Collect data
    data = collector.collect_monte_carlo_data(n_trajectories=30000, dt=0.01, range_noise=5)
    
    # Create dataset
    dataset = MicrogridDataset(data)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Print sample data shape
    sample = next(iter(dataloader))
    print("\nSample data shapes:")
    for key, value in sample.items():
        print(f"{key}: {value.shape}")

    # Train dynamic models
    models, losses = NN_dynamics_training(
        dataloader,
        num_epochs=100,
        learning_rate=2e-4
    )

    # save models
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'pre_train_model/dynamics_model_{i}.pth')

    # Train control models using the same dataset
    control_models, control_losses = NN_control_training(
        dataloader,
        num_epochs=100,
        learning_rate=2e-4
    )

    # Save control models
    for i, model in enumerate(control_models):
        torch.save(model.state_dict(), f'pre_train_model/control_model_{i}.pth')
        
    # Plot control training curves
    plt.figure(figsize=(10, 5))
    for i in range(len(control_losses)):
        plt.plot(control_losses[i], label=f'Controller {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Controller Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_figures/control_training_loss.png')
    plt.close()





        

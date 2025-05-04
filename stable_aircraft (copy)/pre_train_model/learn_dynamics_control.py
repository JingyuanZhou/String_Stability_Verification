import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import from simulate_UAVs.py
from pre_train_model.simulate_UAVs import UAVFormation, FormationVisualizer, PDController, LQRController

# Create output directories if they don't exist
os.makedirs('output_figures', exist_ok=True)

# ==================== Neural Network Models ====================
class DynamicsNN(nn.Module):
    """Neural network for learning UAV dynamics"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DynamicsNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x) + state

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
def collect_training_data(n_uavs=3, controller_type="LQR", n_samples=50000, dt=0.1):
    """Generate training data using existing controller with Monte Carlo sampling
    
    Args:
        n_uavs: Number of UAVs in the formation
        controller_type: Type of controller to use ("PD" or "LQR")
        n_samples: Number of random initial states to sample
        dt: Time step for simulation
        
    Returns:
        Dictionary containing training data
    """
    print(f"Collecting training data using Monte Carlo sampling with {controller_type} controller...")
    
    # Create controller
    if controller_type == "PD":
        controller = PDController(Kp=1.5, Kd=1.0)
    elif controller_type == "LQR":
        controller = LQRController(Q_pos=10.0, Q_vel=1.0, R=1.0)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Create UAV formation
    formation = UAVFormation(n=n_uavs, dim=3, controller=controller)
    
    # Initialize data containers
    states = []
    next_states = []
    actions = []
    follower_states = []
    follower_actions = []
    p_errors = []
    v_errors = []
    
    # Fixed initial state for leader UAV
    leader_position = np.zeros(3)  # Leader at origin
    leader_velocity = np.array([5.0, 0.0, 0.0])  # Leader cruising in x-direction
    
    # Define bounds for random initialization of follower UAVs
    # These are deviations from equilibrium
    pos_error_bound = 6.0  # meters (only in x-direction)
    vel_error_bound = 3.0  # m/s (only in x-direction)
    
    # Generate multiple initial states and collect data
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        # Initialize UAV positions and velocities
        p = np.zeros((n_uavs, 3))
        v = np.zeros((n_uavs, 3))
        
        # Set leader UAV state (fixed)
        p[0] = leader_position
        v[0] = leader_velocity
        
        # Randomly initialize follower UAVs around equilibrium
        for i in range(1, n_uavs):
            # Equilibrium position (behind previous UAV by delta_ref)
            p_eq = p[i-1] - formation.delta_ref
            
            # Add random error only to x-position
            x_error = np.random.uniform(-pos_error_bound, pos_error_bound)
            y_error = np.random.uniform(-pos_error_bound, pos_error_bound)
            z_error = np.random.uniform(-pos_error_bound, pos_error_bound)
            p[i] = p_eq.copy()
            p[i, 0] += x_error  # Apply error only to x-position
            p[i, 1] += y_error  # Apply error only to y-position
            p[i, 2] += z_error  # Apply error only to z-position
            
            # Add random error only to x-velocity
            v_eq = v[i-1].copy()  # Equilibrium velocity matches previous UAV
            x_vel_error = np.random.uniform(-vel_error_bound, vel_error_bound)
            y_vel_error = np.random.uniform(-vel_error_bound, vel_error_bound)
            z_vel_error = np.random.uniform(-vel_error_bound, vel_error_bound)
            v[i] = v_eq.copy()
            v[i, 0] += x_vel_error  # Apply error only to x-velocity
            v[i, 1] += y_vel_error  # Apply error only to y-velocity
            v[i, 2] += z_vel_error  # Apply error only to z-velocity
        
        # Construct state vector
        X_t = np.hstack([p.flatten(), v.flatten()])
        
        # Run a single step of dynamics to get control inputs and next state
        dX_t = formation.dynamics(0.0, X_t)  # Time t=0
        
        # Extract control inputs
        u = np.zeros((n_uavs, 3))
        for i in range(n_uavs):
            u[i] = dX_t[n_uavs*3 + i*3 : n_uavs*3 + (i+1)*3]
        
        # Compute next state using simple Euler integration
        X_next = X_t + dX_t * dt
        
        # Parse next state
        p_next = np.zeros((n_uavs, 3))
        v_next = np.zeros((n_uavs, 3))
        for i in range(n_uavs):
            p_next[i] = X_next[i*3:(i+1)*3]
            v_next[i] = X_next[n_uavs*3 + i*3 : n_uavs*3 + (i+1)*3]
        
        # Store data for all UAVs
        for i in range(n_uavs):
            current_state = np.concatenate([p[i], v[i]])
            next_state = np.concatenate([p_next[i], v_next[i]])
            
            states.append(current_state)
            next_states.append(next_state)
            actions.append(u[i])
            
            # Store extra data for follower UAVs
            if i > 0:
                # Compute reference position and velocity
                p_ref = p[i-1] - formation.delta_ref
                v_ref = v[i-1]
                
                # Compute errors
                p_error = p_ref - p[i]
                v_error = v_ref - v[i]
                
                # Store follower data
                follower_states.append(current_state)
                follower_actions.append(u[i])
                p_errors.append(p_error)
                v_errors.append(v_error)
    
    # Convert to numpy arrays
    states = np.array(states)
    next_states = np.array(next_states)
    actions = np.array(actions)
    follower_states = np.array(follower_states)
    follower_actions = np.array(follower_actions)
    p_errors = np.array(p_errors)
    v_errors = np.array(v_errors)
    
    # Verify shapes
    print(f"Collected data shapes:")
    print(f"All states: {states.shape}")
    print(f"All actions: {actions.shape}")
    print(f"Follower states: {follower_states.shape}")
    print(f"Follower actions: {follower_actions.shape}")
    print(f"Position errors: {p_errors.shape}")
    print(f"Velocity errors: {v_errors.shape}")
    
    # Randomly sample and print an example for debugging
    random_idx = np.random.randint(0, len(states))
    print("\nExample data sample:")
    print(f"Random index: {random_idx}")
    print(f"Current state: {states[random_idx]}")
    print(f"Action: {actions[random_idx]}")
    print(f"Next state: {next_states[random_idx]}")
    
    return {
        'states': states,
        'next_states': next_states,
        'actions': actions,
        'follower_states': follower_states,
        'follower_actions': follower_actions,
        'p_errors': p_errors,
        'v_errors': v_errors
    }

# ==================== Training Functions ====================
def train_dynamics_model(data, epochs=10, batch_size=64, lr=0.0005):
    """Train neural network to learn UAV dynamics"""
    print("Training dynamics model...")
    
    states = torch.FloatTensor(data['states'])
    actions = torch.FloatTensor(data['actions'])
    next_states = torch.FloatTensor(data['next_states'])
    
    # Create dataset and dataloader
    dataset = TensorDataset(states, actions, next_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    model = DynamicsNN(state_dim, action_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    
    # Use tqdm for progress bar
    pbar = tqdm(range(epochs), desc=f"Epoch 0/{epochs}, Loss: {0:.6f}")
    
    for epoch in pbar:
        epoch_loss = 0.0
        for batch_states, batch_actions, batch_next_states in dataloader:
            # Forward pass
            predicted_next_states = model(batch_states, batch_actions)
            loss = criterion(predicted_next_states, batch_next_states)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Update progress bar with current loss
        pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'pre_train_model/dynamics_model.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Dynamics Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('output_figures/dynamics_training_loss.png')
    plt.close()
    
    return model, train_losses

def train_controller_model(data, epochs=10, batch_size=64, lr=0.0005):
    """Train neural network to learn UAV controller"""
    print("Training controller model...")
    
    p_errors = torch.FloatTensor(data['p_errors'])
    v_errors = torch.FloatTensor(data['v_errors'])
    actions = torch.FloatTensor(data['follower_actions'])  # Only follower UAV actions
    
    # Create concatenated error state
    error_states = torch.cat([p_errors, v_errors], dim=1)
    
    # Create dataset and dataloader
    print(f"Error states shape: {error_states.shape}")
    print(f"Actions shape: {actions.shape}")
    
    dataset = TensorDataset(error_states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = error_states.shape[1]  # Position error + velocity error
    output_dim = actions.shape[1]
    model = ControllerNN(input_dim, output_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    
    # Use tqdm for progress bar
    pbar = tqdm(range(epochs), desc=f"Epoch 0/{epochs}, Loss: {0:.6f}")
    
    for epoch in pbar:
        epoch_loss = 0.0
        for batch_error_states, batch_actions in dataloader:
            # Forward pass
            predicted_actions = model(batch_error_states)
            loss = criterion(predicted_actions, batch_actions)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Update progress bar with current loss
        pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), 'pre_train_model/controller_model.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Controller Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('output_figures/controller_training_loss.png')
    plt.close()
    
    return model, train_losses

# ==================== Neural Network Controller Class ====================
class NNController:
    """Neural network controller that wraps the trained model"""
    def __init__(self, model_path):
        # Load model architecture
        input_dim = 6  # 3D position error + 3D velocity error
        output_dim = 3  # 3D control input
        self.model = ControllerNN(input_dim, output_dim)
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def compute_control(self, p_ref, p, v_ref, v):
        # Convert numpy arrays to torch tensors
        p_error = torch.FloatTensor(p_ref - p).unsqueeze(0)
        v_error = torch.FloatTensor(v_ref - v).unsqueeze(0)
        
        x_error = torch.cat([p_error, v_error], dim=1)
        # Compute control action
        with torch.no_grad():
            u = self.model(x_error).squeeze(0).numpy()
        
        return u

# ==================== Main Function ====================
def main():
    # Configuration
    training = True
    n_uavs = 5
    if training:
        controller_type = "LQR"  # Use PD controller to generate training data
        epochs = 10
    
        # Collect data
        data = collect_training_data(n_uavs=n_uavs, controller_type=controller_type)
    
    # Train dynamics model
        dynamics_model, dynamics_losses = train_dynamics_model(data, epochs=epochs)
    
    # Train controller model
        controller_model, controller_losses = train_controller_model(data, epochs=epochs)
    
    # Evaluate models
    #dynamics_mse = evaluate_dynamics_model(dynamics_model, data)
    #controller_mse = evaluate_controller_model(controller_model, data)
    
    # Test learned controller in simulation
    print("\nTesting learned neural network controller in simulation...")
    nn_controller = NNController('pre_train_model/controller_model.pth')
    
    # Create formation with neural network controller
    formation = UAVFormation(n=n_uavs, dim=3, controller=nn_controller)
    formation.run_simulation()
    
    # Visualize results
    visualizer = FormationVisualizer(formation)
    visualizer.plot_complete_trajectory(controller_name="NeuralNetwork")
    ani = visualizer.animate_formation(controller_name="NeuralNetwork")
    
    print("\nTraining and evaluation complete!")
    print("Results saved in 'output_figures/' directory")
    print("Trained models saved in 'trained_models/' directory")
    
    return formation, visualizer, ani

if __name__ == "__main__":
    formation, visualizer, ani = main()

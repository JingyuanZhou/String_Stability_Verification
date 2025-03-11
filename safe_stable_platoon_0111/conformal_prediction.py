import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from networks import NetworkController
import os
import torch.optim as optim

def train_dynamics_model(model, X, X_control, Y, cav_idx, cav_indices, batch_size=32, learning_rate=0.001, model_name="dynamics_model", epochs=100):
    """
    Train the dynamics model for a specific CAV.
    
    Args:
        model: Neural network model to train
        X: Input states (batch_size, state_dim)
        X_control: Control inputs (batch_size, control_dim)
        Y: Target next states (batch_size, state_dim)
        cav_idx: Index of the CAV this network is for
        cav_indices: List of CAV indices to know which control to use
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        model_name: Name for saving the model
        epochs: Number of training epochs
    """
    # Extract the target values for this CAV (spacing and velocity)
    spacing_idx = 2 * cav_idx
    velocity_idx = 2 * cav_idx + 1
    
    # Find which position in the control vector corresponds to this CAV
    cav_model_idx = next((i for i, idx in enumerate(cav_indices) if idx == cav_idx), 0)
    cav_control = X_control[:, cav_model_idx:cav_model_idx+1]
    
    # The target is just the spacing and velocity for this CAV
    cav_target = Y[:, spacing_idx:spacing_idx+2]
    
    # Create a dataset for training
    train_dataset = TensorDataset(X, cav_control, cav_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_control, batch_Y in train_loader:
            optimizer.zero_grad()
            
            # Concatenate state and control
            inputs = torch.cat([batch_X, batch_control], dim=1)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, batch_Y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"CAV {cav_idx} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}.pth")
    print(f"Model saved to models/{model_name}.pth")
    
    return model

class PlatoonDynamics(nn.Module):
    """Neural network to approximate platoon dynamics with pre-trained controllers."""
    def __init__(self, num_vehicles=4, cav_indices=[1, 2, 3], hidden_dim=64):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.cav_indices = cav_indices
        
        # State dimensions: [s0, v0, s1, v1, s2, v2, s3, v3]
        # where s = spacing, v = velocity, and the number is the vehicle index
        self.state_dim = 2 * num_vehicles  # alternating spacing and velocity for each vehicle
        self.control_dim = 1  # one control input for each CAV
        
        # Create a separate neural network for each CAV's dynamics
        self.dynamics_networks = nn.ModuleList()
        for _ in range(len(cav_indices)):
            network = nn.Sequential(
                nn.Linear(self.state_dim + self.control_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
            self.dynamics_networks.append(network)
        
        # Pre-trained controllers (will be loaded separately)
        self.controllers = [None] * num_vehicles
    
    def load_controllers(self, pre_trained_id=99, device='cpu'):
        """Load pre-trained controllers for CAVs."""
        for i, cav_idx in enumerate(self.cav_indices):
            controller_path = f"pre_train/pre_train_model_marl/sac_platoon_{pre_trained_id}_actor_{i}.pth"

            # Load controller parameters
            raw_parameters = torch.load(controller_path, map_location=device)
            controller = NetworkController(self.state_dim, 1).to(device)
            
            # Process parameters to match controller network structure
            controller_parameters = {}
            for k, v in raw_parameters.items():
                if k.startswith('trunk'):
                    new_key = k.replace('trunk', 'network')
                    # Only keep mean output from the last layer
                    if 'network.4.weight' in new_key:
                        controller_parameters[new_key] = v[:1, :]
                    elif 'network.4.bias' in new_key:
                        controller_parameters[new_key] = v[:1]
                    else:
                        controller_parameters[new_key] = v
            
            controller.load_state_dict(controller_parameters)
            self.controllers[cav_idx] = controller
            print(f"Loaded pre-trained model for CAV {cav_idx}")
    
    def compute_control_actions(self, state, batched=True, batch_size=256):
        """Compute control actions for all vehicles using pre-trained controllers."""
        # Ensure state has batch dimension
        if not batched and state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension for processing
        
        # Create control vector (acceleration for each vehicle)
        if batched:
            batch_size_actual = state.shape[0]
            control = torch.zeros(batch_size_actual, self.num_vehicles, device=state.device)
        else:
            control = torch.zeros(1, self.num_vehicles, device=state.device)
        
        # Process in batches to avoid memory issues
        for start_idx in range(0, state.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, state.shape[0])
            batch_states = state[start_idx:end_idx]
            batch_size_actual = batch_states.shape[0]
            batch_x_star = torch.zeros(batch_size_actual, 8, dtype=torch.float32)
            batch_x_star[:, 0::2] = 20.0  # Spacing references
            batch_x_star[:, 1::2] = 15.0  # Velocity references
            
            # Apply controllers for CAVs
            for i, cav_idx in enumerate(self.cav_indices):
                if self.controllers[cav_idx] is not None:
                    # Pass the FULL state to the controller
                    # The pre-trained controllers expect the complete state vector
                    control_output = self.controllers[cav_idx](
                        batch_states,  # Use full state here
                        batch_x_star,  # Use full state here
                        None,  # No reference control
                        None   # No bounds
                    ).squeeze(-1)
                    
                    # Store the control value
                    control[start_idx:end_idx, cav_idx] = control_output
                else:
                    # Fallback to simple control law for this CAV
                    for b in range(batch_size_actual):
                        # Get current spacing and velocity
                        spacing_idx = 2 * cav_idx
                        velocity_idx = 2 * cav_idx + 1
                        
                        # Skip if beyond the state dimension
                        if spacing_idx >= batch_states.shape[1] or velocity_idx >= batch_states.shape[1]:
                            print(f"Warning: CAV {cav_idx} exceeds state dimension {batch_states.shape[1]}")
                            continue
                        
                        spacing = batch_states[b, spacing_idx]
                        velocity = batch_states[b, velocity_idx]
                        
                        # Simple spacing and velocity error based control
                        spacing_error = spacing - 20.0  # Desired spacing
                        velocity_error = velocity - 15.0  # Desired velocity
                        
                        # Compute control: -K[s - s_des, v - v_des]
                        # Negative feedback control
                        control_value = -0.5 * spacing_error - 0.7 * velocity_error
                        
                        # Apply control limits
                        control_value = max(min(control_value, 5.0), -5.0)
                        
                        # Store control for this CAV
                        control[start_idx + b, cav_idx] = control_value
            
            # Apply control for vehicle 0 (lead vehicle)
            for b in range(batch_size_actual):
                # Lead vehicle follows a simple desired speed profile
                velocity = batch_states[b, 1]  # v0
                
                # Maintain speed around 15 m/s with small oscillations
                desired_velocity = 15.0 + 2.0 * torch.sin(torch.tensor(0.1 * start_idx))
                velocity_error = velocity - desired_velocity
                
                # Simple proportional control
                control_value = -0.5 * velocity_error
                
                # Apply control limits
                control_value = max(min(control_value, 3.0), -3.0)
                
                # Store control for lead vehicle
                control[start_idx + b, 0] = control_value
            
            # Apply control for human-driven vehicles (if any)
            for v in range(1, self.num_vehicles):
                if v not in self.cav_indices:  # This is an HDV
                    for b in range(batch_size_actual):
                        # Skip if beyond the state dimension
                        spacing_idx = 2 * v
                        velocity_idx = 2 * v + 1
                        prev_velocity_idx = 2 * (v-1) + 1
                        
                        if (spacing_idx >= batch_states.shape[1] or 
                            velocity_idx >= batch_states.shape[1] or 
                            prev_velocity_idx >= batch_states.shape[1]):
                            continue
                        
                        # Get vehicle state
                        spacing = batch_states[b, spacing_idx]
                        velocity = batch_states[b, velocity_idx]
                        
                        # Calculate optimal velocity based on current spacing
                        optimal_velocity = 5.0 + 15.0 * (torch.tanh(0.5 * (spacing - 15.0)) + torch.tanh(torch.tensor(0.5)))
                        
                        # Control is acceleration towards optimal velocity
                        control_value = 0.4 * (optimal_velocity - velocity)
                        
                        # Apply control limits
                        control_value = max(min(control_value, 3.0), -3.0)
                        
                        # Store control for this HDV
                        control[start_idx + b, v] = control_value
        
        # Remove batch dimension if not needed
        if not batched:
            return control.squeeze(0)
        
        return control
    
    def forward(self, state, control, dynamics_model_idx=[0,1,2]):
        """
        Forward pass through the dynamics network.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            control: Tensor of shape (batch_size, control_dim)
            dynamics_model_idx: Index of which dynamics model to use (default: 0)
        
        Returns:
            Predicted next state
        """
        # Concatenate state and control
        for i in dynamics_model_idx:
            x = torch.cat([state, control[:,i].unsqueeze(-1)], dim=-1)
            x = self.dynamics_networks[i](x)
            if i == 0:
                x_pred = x
            else:
                x_pred = torch.cat([x_pred, x], dim=-1)
        
        return x_pred
        
        # Use the specified dynamics model
        
    
    def predict_next_state(self, state, control=None, dynamics_model_idx=0):
        """
        Predict next state using the neural dynamics.
        
        If control is not provided, it will be computed using pre-trained controllers
        or fallback to simple control laws.
        """
        if control is None:
            # For a single state input
            if state.dim() == 1:
                control_full = self.compute_control_actions(state, batched=False)
                # Extract only the CAV controls
                control = torch.zeros(self.control_dim, device=state.device)
                for i, cav_idx in enumerate(self.cav_indices):
                    control[i] = control_full[cav_idx]
            else:
                # For batched state input
                control_full = self.compute_control_actions(state, batched=True)
                # Extract only the CAV controls
                batch_size = state.shape[0]
                control = torch.zeros(batch_size, self.control_dim, device=state.device)
                for i, cav_idx in enumerate(self.cav_indices):
                    control[:, i] = control_full[:, cav_idx]
        
        return self.forward(state, control, dynamics_model_idx)

def generate_platoon_data(num_vehicles=4, cav_indices=[1, 2, 3], n_samples=1000, noise_level=0.1, pre_trained_id=99, batch_size=64):
    """Generate data for a platoon with pre-trained controllers."""
    # State dimension: alternating spacing and velocity for each vehicle [s0, v0, s1, v1, ...]
    state_dim = 2 * num_vehicles
    
    # Load pre-trained controllers
    controllers = [None] * num_vehicles
    for i, cav_idx in enumerate(cav_indices):
        controller_path = f"pre_train/pre_train_model_marl/sac_platoon_{pre_trained_id}_actor_{i}.pth"
        # Load controller parameters
        raw_parameters = torch.load(controller_path, map_location='cpu')
        controller = NetworkController(state_dim, 1)
        
        # Process parameters to match controller network structure
        controller_parameters = {}
        for k, v in raw_parameters.items():
            if k.startswith('trunk'):
                new_key = k.replace('trunk', 'network')
                # Only keep mean output from the last layer
                if 'network.4.weight' in new_key:
                    controller_parameters[new_key] = v[:1, :]
                elif 'network.4.bias' in new_key:
                    controller_parameters[new_key] = v[:1]
                else:
                    controller_parameters[new_key] = v
        
        controller.load_state_dict(controller_parameters)
        controllers[cav_idx] = controller
        print(f"Loaded pre-trained model for CAV {cav_idx}")
    
    # Initialize data arrays
    states = np.zeros((n_samples, state_dim))
    next_states = np.zeros((n_samples, state_dim))
    controls = np.zeros((n_samples, num_vehicles))  # Control inputs for each vehicle
    
    # Set initial state distribution
    for i in range(num_vehicles):
        # Spacing - even indices (0, 2, 4, 6)
        if i == 0:
            # Lead vehicle spacing (distance to fixed reference point or virtual leader)
            states[:, 2*i] = np.ones(n_samples) * 20 # 20
        else:
            # Following vehicles spacing
            states[:, 2*i] = np.random.uniform(15, 25, n_samples)  # spacing between 15-25 m
        
        # Velocity - odd indices (1, 3, 5, 7)
        states[:, 2*i+1] = np.random.uniform(10, 20, n_samples)  # velocity between 10-20 m/s
    
    # Simple dynamics for demonstration
    dt = 0.1  # seconds
    
    # Process in batches to avoid memory issues
    states_tensor = torch.tensor(states, dtype=torch.float32)
    
    # Compute control for CAVs using pre-trained controllers in batches
    with torch.no_grad():
        for cav_idx in cav_indices:
            if controllers[cav_idx] is not None:
                # Process in batches
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_states = states_tensor[start_idx:end_idx]
                    batch_size_actual = batch_states.shape[0]
                    
                    # Create reference states where even indices (spacing) = 20.0, odd indices (velocity) = 15.0
                    batch_x_star = torch.zeros(batch_size_actual, 8, dtype=torch.float32)
                    # Set all even indices (0,2,4,6) to desired spacing (20.0)
                    batch_x_star[:, 0::2] = 20.0  # Spacing references
                    # Set all odd indices (1,3,5,7) to desired velocity (15.0)
                    batch_x_star[:, 1::2] = 15.0  # Velocity references
                    
                    batch_u_star = torch.zeros(batch_size_actual, 1, dtype=torch.float32)
                    batch_u_min = torch.full((batch_size_actual, 1), -5.0, dtype=torch.float32)
                    batch_u_max = torch.full((batch_size_actual, 1), 5.0, dtype=torch.float32)
                    
                    # Get control actions for this batch
                    batch_control = controllers[cav_idx](
                        batch_states, 
                        batch_x_star,
                        batch_u_star, 
                        (batch_u_min, batch_u_max)
                    ).cpu().numpy().squeeze()
                    
                    if batch_size_actual == 1:
                        controls[start_idx, cav_idx] = batch_control
                    else:
                        controls[start_idx:end_idx, cav_idx] = batch_control
            
            else:
                # No controller available, use simple control law
                for i in range(n_samples):
                    spacing_idx = 2 * cav_idx  # Index for spacing
                    velocity_idx = 2 * cav_idx + 1  # Index for velocity
                    prev_velocity_idx = 2 * (cav_idx-1) + 1  # Index for predecessor's velocity
                    
                    spacing_error = states[i, spacing_idx] - 20.0
                    velocity_error = states[i, prev_velocity_idx] - states[i, velocity_idx]
                    
                    controls[i, cav_idx] = 0.5 * spacing_error + 0.7 * velocity_error
                    controls[i, cav_idx] = np.clip(controls[i, cav_idx], -3.0, 3.0)
    
    # Apply dynamics and compute next states
    for i in range(n_samples):
        # Copy initial state to next state
        next_states[i] = states[i].copy()
        
        # Head vehicle (lead vehicle): apply random acceleration
        controls[i, 0] = np.random.normal(0, 0.5)
        next_states[i, 1] = states[i, 1] + controls[i, 0] * dt  # Update velocity v0
        # Spacing for lead vehicle remains unchanged since it's relative to a fixed point
        
        # Following vehicles: apply control and update states
        for v in range(1, num_vehicles):
            if v not in cav_indices:
                # Human-Driven Vehicle (HDV) uses OVM-like model
                spacing_idx = 2 * v  # Current vehicle's spacing index
                velocity_idx = 2 * v + 1  # Current vehicle's velocity index
                
                spacing = states[i, spacing_idx]
                velocity = states[i, velocity_idx]
                
                # Calculate optimal velocity based on current spacing
                optimal_velocity = 5.0 + 15.0 * (np.tanh(0.5 * (spacing - 15.0)) + np.tanh(0.5))
                # Control is acceleration towards optimal velocity
                controls[i, v] = 0.4 * (optimal_velocity - velocity)
            
            # Update velocity for all following vehicles (both CAVs and HDVs)
            velocity_idx = 2 * v + 1
            next_states[i, velocity_idx] = states[i, velocity_idx] + controls[i, v] * dt
            
            # Update spacing: s_i(t+1) = s_i(t) + [v_{i-1}(t) - v_i(t)] * dt
            spacing_idx = 2 * v
            prev_velocity_idx = 2 * (v-1) + 1
            next_states[i, spacing_idx] = states[i, spacing_idx] + (states[i, prev_velocity_idx] - states[i, velocity_idx]) * dt
    
    # Add noise to all next states
    next_states += np.random.normal(0, noise_level, next_states.shape)
    
    # Ensure physical constraints
    for v in range(num_vehicles):
        velocity_idx = 2 * v + 1
        spacing_idx = 2 * v
        # Non-negative velocity
        next_states[:, velocity_idx] = np.maximum(next_states[:, velocity_idx], 0)
        # Minimum spacing
        if v > 0:  # Only constrain spacing for following vehicles
            next_states[:, spacing_idx] = np.maximum(next_states[:, spacing_idx], 5)
    
    # Extract controls for CAVs only (for supervised learning)
    cav_controls = np.zeros((n_samples, len(cav_indices)))
    for i, cav_idx in enumerate(cav_indices):
        cav_controls[:, i] = controls[:, cav_idx]
    
    return (torch.tensor(states, dtype=torch.float32), 
            torch.tensor(cav_controls, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32))

class EnsembleConformalForecaster:
    """
    Conformal prediction for platoon dynamics.
    Handles multiple CAV dynamics models.
    """
    def __init__(self, model, alpha=0.1, state_dim=8, control_dim=3, cav_indices=[1, 2, 3]):
        self.model = model
        self.alpha = alpha
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.cav_indices = cav_indices
        self.num_models = len(cav_indices)
        
        # Separate quantiles for each CAV and each dimension (spacing and velocity)
        self.quantiles = {}
        for i, cav_idx in enumerate(cav_indices):
            self.quantiles[cav_idx] = {"spacing": None, "velocity": None}
        
    def compute_residuals(self, X, U, Y):
        """Compute residuals between predictions and actual next states for all CAVs."""
        residuals = {}
        for i, cav_idx in enumerate(self.cav_indices):
            spacing_idx = 2 * cav_idx
            velocity_idx = 2 * cav_idx + 1
            
            with torch.no_grad():
                # Get model index for this CAV
                model_idx = i
                
                # Make predictions using the specified CAV's dynamics model
                cav_control = U[:, i:i+1]  # Extract just this CAV's control
                
                # Forward pass through the network
                inputs = torch.cat([X, cav_control], dim=1)
                preds = self.model.dynamics_networks[model_idx](inputs)
                
                # Calculate absolute errors for spacing and velocity separately
                residuals[cav_idx] = {
                    "spacing": torch.abs(preds[:, 0] - Y[:, spacing_idx]),
                    "velocity": torch.abs(preds[:, 1] - Y[:, velocity_idx])
                }
        
        return residuals
    
    def fit(self, X_cal, U_cal, Y_cal):
        """Fit the conformal predictor using calibration data."""
        residuals = self.compute_residuals(X_cal, U_cal, Y_cal)
        
        # Compute quantiles for each CAV and dimension
        level = (1 - self.alpha) * (1 + 1/len(X_cal))
        
        for cav_idx in self.cav_indices:
            # Get spacing quantile
            spacing_residuals = residuals[cav_idx]["spacing"]
            self.quantiles[cav_idx]["spacing"] = torch.quantile(spacing_residuals, level)
            
            # Get velocity quantile
            velocity_residuals = residuals[cav_idx]["velocity"]
            self.quantiles[cav_idx]["velocity"] = torch.quantile(velocity_residuals, level)
            
            print(f"CAV {cav_idx} calibration quantiles - Spacing: {self.quantiles[cav_idx]['spacing']:.4f}, Velocity: {self.quantiles[cav_idx]['velocity']:.4f}")
    
    def predict_with_bounds(self, X, U, model_indices=None):
        """
        Make predictions with uncertainty bounds.
        
        Args:
            X: Input state tensor (batch_size, state_dim) or (state_dim)
            U: Control input tensor (batch_size, control_dim) or (control_dim)
            model_indices: Which models to use for prediction (default: all)
            
        Returns:
            mean: Mean prediction (batch_size, state_dim)
            lower: Lower bound (batch_size, state_dim)
            upper: Upper bound (batch_size, state_dim)
        """
        if model_indices is None:
            model_indices = list(range(self.num_models))
            
        # Ensure X and U have batch dimension
            
        batch_size = X.shape[0]
            
        # Create output tensors starting with the input state

        X = X.squeeze().unsqueeze(0)
        U = U.squeeze().unsqueeze(0)
        next_state = X.clone()
        lower_bound = X.clone()
        upper_bound = X.clone()
        
        with torch.no_grad():
            # Update predictions for each CAV
            for i in model_indices:
                cav_idx = self.cav_indices[i]
                spacing_idx = 2 * cav_idx
                velocity_idx = 2 * cav_idx + 1
                
                # Get control for this CAV
                cav_control = U[:, i:i+1]
                
                # Concatenate state and control for this CAV
                inputs = torch.cat([X, cav_control], dim=1)
                
                # Get prediction for this CAV's spacing and velocity
                pred = self.model.dynamics_networks[i](inputs)
                
                # Update the mean prediction in the output tensor
                next_state[:, spacing_idx] = pred[:, 0]
                next_state[:, velocity_idx] = pred[:, 1]
                
                # Set bounds using the calibrated quantiles
                if cav_idx in self.quantiles and self.quantiles[cav_idx]["spacing"] is not None:
                    # Spacing bounds
                    lower_bound[:, spacing_idx] = pred[:, 0] - self.quantiles[cav_idx]["spacing"]
                    upper_bound[:, spacing_idx] = pred[:, 0] + self.quantiles[cav_idx]["spacing"]
                    
                    # Velocity bounds
                    lower_bound[:, velocity_idx] = pred[:, 1] - self.quantiles[cav_idx]["velocity"]
                    upper_bound[:, velocity_idx] = pred[:, 1] + self.quantiles[cav_idx]["velocity"]
                else:
                    # If not calibrated, use a default uncertainty of 10%
                    lower_bound[:, spacing_idx] = pred[:, 0] * 0.9
                    upper_bound[:, spacing_idx] = pred[:, 0] * 1.1
                    lower_bound[:, velocity_idx] = pred[:, 1] * 0.9
                    upper_bound[:, velocity_idx] = pred[:, 1] * 1.1
            
            # Non-CAV vehicles (lead vehicle and any HDVs) just copy from the input
            # Update lead vehicle (vehicle 0) using simple constant velocity model
            if 0 not in self.cav_indices:
                v0 = X[:, 1]  # Lead vehicle velocity
                s0 = X[:, 0]  # Lead vehicle spacing
                
                # Simple model: Constant velocity for lead vehicle
                dt = 0.1  # Time step
                next_state[:, 1] = v0  # Velocity stays the same
                next_state[:, 0] = s0 + v0 * dt  # Spacing increases by velocity * time step
                
                # No uncertainty for lead vehicle
                lower_bound[:, 0] = next_state[:, 0]
                lower_bound[:, 1] = next_state[:, 1]
                upper_bound[:, 0] = next_state[:, 0]
                upper_bound[:, 1] = next_state[:, 1]
                
                # Also update any human-driven vehicles (HDVs)
                for v in range(1, self.state_dim // 2):
                    if v not in self.cav_indices:  # If this is not a CAV
                        spacing_idx = 2 * v
                        velocity_idx = 2 * v + 1
                        prev_velocity_idx = 2 * (v-1) + 1
                        
                        # HDV model: OVM-like model
                        spacing = X[:, spacing_idx]
                        velocity = X[:, velocity_idx]
                        prev_velocity = X[:, prev_velocity_idx]
                        
                        # Calculate optimal velocity based on current spacing
                        optimal_velocity = 5.0 + 15.0 * (torch.tanh(0.5 * (spacing - 15.0)) + torch.tanh(torch.tensor(0.5)))
                        # Control is acceleration towards optimal velocity
                        hdv_control = 0.4 * (optimal_velocity - velocity)
                        
                        # Update velocity
                        next_velocity = velocity + hdv_control * 0.1  # dt = 0.1
                        next_state[:, velocity_idx] = next_velocity
                        
                        # Update spacing: s_i(t+1) = s_i(t) + [v_{i-1}(t) - v_i(t)] * dt
                        next_spacing = spacing + (prev_velocity - velocity) * 0.1  # dt = 0.1
                        next_state[:, spacing_idx] = next_spacing
                        
                        # No uncertainty for HDVs
                        lower_bound[:, spacing_idx] = next_state[:, spacing_idx]
                        lower_bound[:, velocity_idx] = next_state[:, velocity_idx]
                        upper_bound[:, spacing_idx] = next_state[:, spacing_idx]
                        upper_bound[:, velocity_idx] = next_state[:, velocity_idx]
                    
        return next_state, lower_bound, upper_bound
    
    def evaluate(self, X_test, U_test, Y_test):
        """Evaluate the conformal predictor on test data."""
        # Count how many test points fall within the prediction intervals
        in_interval = 0
        total_points = len(X_test) * len(self.cav_indices) * 2  # Total CAV predictions (spacing and velocity for each)
        
        # Process in batches or one by one
        for i in range(len(X_test)):
            # Make sure inputs have batch dimension
            x_i = X_test[i].unsqueeze(0) if X_test[i].dim() == 1 else X_test[i]
            u_i = U_test[i].unsqueeze(0) if U_test[i].dim() == 1 else U_test[i]
            
            # Make prediction with bounds
            mean, lower, upper = self.predict_with_bounds(x_i, u_i)
            
            # Remove batch dimension for comparison
            mean = mean.squeeze(0)
            lower = lower.squeeze(0)
            upper = upper.squeeze(0)
            
            # Check if actual next state is within bounds for each CAV
            for j, cav_idx in enumerate(self.cav_indices):
                spacing_idx = 2 * cav_idx
                velocity_idx = 2 * cav_idx + 1
                
                # Check spacing
                if lower[spacing_idx].item() <= Y_test[i, spacing_idx].item() <= upper[spacing_idx].item():
                    in_interval += 1
                
                # Check velocity
                if lower[velocity_idx].item() <= Y_test[i, velocity_idx].item() <= upper[velocity_idx].item():
                    in_interval += 1
        
        return in_interval / total_points

def run_platoon_conformal_prediction():
    """Run conformal prediction for a platoon with pre-trained controllers."""
    # Define platoon parameters
    num_vehicles = 4
    cav_indices = [1, 2, 3]  # Indices of Connected Automated Vehicles (CAVs)
    
    # Generate data
    print("Generating platoon data...")
    X, X_control, Y = generate_platoon_data(num_vehicles=num_vehicles, cav_indices=cav_indices, n_samples=3000, noise_level=0.2)
    
    # Split data into train, calibration, and test sets
    print("Splitting data...")
    n_samples = X.shape[0]
    train_size = int(0.7 * n_samples)
    cal_size = int(0.15 * n_samples)
    
    # Training data
    X_train = X[:train_size]
    X_control_train = X_control[:train_size]
    Y_train = Y[:train_size]
    
    # Calibration data
    X_cal = X[train_size:train_size+cal_size]
    X_control_cal = X_control[train_size:train_size+cal_size]
    Y_cal = Y[train_size:train_size+cal_size]
    
    # Test data
    X_test = X[train_size+cal_size:]
    X_control_test = X_control[train_size+cal_size:]
    Y_test = Y[train_size+cal_size:]
    
    # Train dynamics model (different model for each CAV's dynamics)
    print("Training dynamics model...")
    model = PlatoonDynamics(num_vehicles=num_vehicles, cav_indices=cav_indices, hidden_dim=128)
    
    # Train each dynamics model separately
    for i, cav_idx in enumerate(cav_indices):
        train_dynamics_model(
            model=model.dynamics_networks[i],
            X=X_train, 
            X_control=X_control_train, 
            Y=Y_train,
            cav_idx=cav_idx,
            cav_indices=cav_indices,
            batch_size=64,
            learning_rate=0.001,
            model_name=f"dynamics_model_cav{cav_idx}",
            epochs=50
        )
    
    # Initialize and fit conformal predictor
    print("Fitting conformal predictor...")
    predictor = EnsembleConformalForecaster(
        model=model,
        alpha=0.1,  # 90% prediction intervals
        state_dim=2 * num_vehicles,
        control_dim=len(cav_indices),
        cav_indices=cav_indices
    )
    
    # Fit using calibration data
    predictor.fit(X_cal, X_control_cal, Y_cal)
    
    # Evaluate on test data
    print("Evaluating on test data...")
    coverage = predictor.evaluate(X_test, X_control_test, Y_test)
    print(f"Empirical coverage: {coverage:.4f}")
    
    # Generate a trajectory with uncertainty bounds
    print("Generating trajectory with uncertainty bounds...")
    # Try to load controllers if available (used to generate control inputs)
    try:
        model.load_controllers(pre_trained_id=99)
    except Exception as e:
        print(f"Could not load controllers: {e}")
        print("Will use simple control laws instead")
    
    # Initial state [s0, v0, s1, v1, s2, v2, s3, v3]
    initial_state = torch.tensor([
        20.0,  # s0: Lead spacing (to virtual reference)
        13.0,  # v0: Lead velocity
        21.0,  # s1: CAV1 spacing
        16.0,  # v1: CAV1 velocity
        18.0,  # s2: CAV2 spacing
        14.0,  # v2: CAV2 velocity
        19.0,  # s3: CAV3 spacing
        15.0,  # v3: CAV3 velocity
    ], dtype=torch.float32)
    
    # Number of steps to predict
    steps = 100
    
    # Storage for predictions
    states = [initial_state.numpy().squeeze().tolist()]
    means = [initial_state.numpy().squeeze().tolist()]
    lowers = []
    uppers = []
    all_controls = []
    
    # Generate the trajectory
    state = initial_state
    for t in range(steps):
        # Extract controls from all CAVs using the model's controllers
        full_controls = model.compute_control_actions(state, batched=False)
        
        # Extract just the CAV controls
        controls = torch.zeros(len(cav_indices), dtype=torch.float32)
        for i, cav_idx in enumerate(cav_indices):
            controls[i] = full_controls[cav_idx]
        
        all_controls.append(controls.detach().numpy())
        
        # Get prediction with uncertainty bounds
        mean, lower, upper = predictor.predict_with_bounds(state.unsqueeze(0), controls.unsqueeze(0))
        
        # Update state for next step (use the mean prediction)
        state = mean
        
        # Store predictions
        states.append(state.numpy().squeeze().tolist())
        means.append(mean.numpy().squeeze().tolist())
        lowers.append(lower.numpy().squeeze().tolist())
        uppers.append(upper.numpy().squeeze().tolist())
    
    # Convert to arrays
    means = np.array(means)
    lowers = np.array(lowers)
    uppers = np.array(uppers)
    all_controls = np.array(all_controls)
    
    # Plot trajectory - show velocities for all vehicles
    fig, axs = plt.subplots(num_vehicles, 1, figsize=(12, 12))
    time = [i * 0.1 for i in range(steps+1)]
    
    # Plot velocities (odd indices: 1, 3, 5, 7)
    for i in range(num_vehicles):
        ax = axs[i]
        velocity_idx = 2*i + 1  # Velocity index for vehicle 
        ax.plot(time, means[:, velocity_idx], 'b-', label='Predicted Mean')
        ax.fill_between(time[1:], lowers[:, velocity_idx], uppers[:, velocity_idx], color='b', alpha=0.2, 
                        label=f'{(1-predictor.alpha)*100:.0f}% PI')
        ax.set_ylabel(f'Vehicle {i} Velocity (m/s)')
        ax.legend()
        ax.grid(True)
    
    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('Platoon Velocity Predictions with Uncertainty')
    plt.tight_layout()
    plt.savefig('output_figures/platoon_velocity_predictions.png')
    plt.close()
    
    # Plot spacings (even indices: 0, 2, 4, 6)
    fig, axs = plt.subplots(num_vehicles, 1, figsize=(12, 12))
    
    for i in range(num_vehicles):
        spacing_idx = 2*i  # Spacing index for vehicle i
        ax = axs[i]
        ax.plot(time, means[:, spacing_idx], 'r-', label='Predicted Mean')
        ax.fill_between(time[1:], lowers[:, spacing_idx], uppers[:, spacing_idx], color='r', alpha=0.2, 
                        label=f'{(1-predictor.alpha)*100:.0f}% PI')
        ax.set_ylabel(f'Vehicle {i} Spacing (m)')
        ax.legend()
        ax.grid(True)
    
    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('Platoon Spacing Predictions with Uncertainty')
    plt.tight_layout()
    plt.savefig('output_figures/platoon_spacing_predictions.png')
    
    # Plot control inputs
    fig, axs = plt.subplots(len(cav_indices), 1, figsize=(12, 9))
    time_controls = np.arange(0, steps*0.1, 0.1)
    
    for i, cav_idx in enumerate(cav_indices):
        ax = axs[i] if len(cav_indices) > 1 else axs
        ax.plot(time_controls, all_controls[:, i], 'g-')
        ax.set_ylabel(f'CAV {cav_idx} Control (m/s²)')
        ax.grid(True)
    
    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('Control Inputs for CAVs')
    plt.tight_layout()
    plt.savefig('output_figures/platoon_control_inputs.png')
    
    print("Conformal prediction for platoon completed! Check the output figures.")
    
    return predictor, model

if __name__ == "__main__":
    # Use the function to run the example
    run_platoon_conformal_prediction()

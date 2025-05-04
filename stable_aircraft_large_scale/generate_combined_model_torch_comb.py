import torch
import torch.nn as nn
import torch.onnx
from onnxsim import simplify
import onnx
import numpy as np
from networks import VectorLyapunovNetwork, ControllerNN, CombinedController
from pre_train_model.learn_dynamics_control import DynamicsNN

class CombinedUAVNetwork(nn.Module):
    def __init__(self, controller, V_net, controlled_indices, state_dims, system_dynamics):
        super(CombinedUAVNetwork, self).__init__()
        self.controller = controller
        self.V_net = V_net
        self.system_dynamics = system_dynamics
        self.controlled_indices = controlled_indices
        self.state_dims = state_dims
        self.num_uavs = len(state_dims)
        self.dim = state_dims[0] // 2  # Each UAV has position and velocity
        self.dt = 0.1
        
        # Control bounds
        self.max_thrust = 5.0
        self.min_thrust = -5.0
        
        # Reference position between UAVs (default value from dynamics params)
        self.delta_ref = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32, requires_grad=False).unsqueeze(0)

        self._build_selection_matrices()

    def _build_selection_matrices(self):
        """
        Build selection matrices for extracting UAV states and calculating state errors.
        Support dynamic number of UAVs in a string formation.
        """
        dim = self.dim
        state_dim = 2 * dim  # Position + velocity dimensions
        total_size = self.num_uavs * state_dim
        
        # Create selection matrices for each UAV
        self.uav_selection_matrices = []
        for i in range(self.num_uavs):
            # Matrix to select state of UAV i from flattened state vector
            selection_matrix = torch.zeros(total_size, state_dim)
            start_idx = i * state_dim
            end_idx = (i + 1) * state_dim
            selection_matrix[start_idx:end_idx, :] = torch.eye(state_dim)
            self.uav_selection_matrices.append(selection_matrix)
        
        # Create error matrices for followers
        self.error_matrices = []
        for i in range(1, self.num_uavs):  # For each follower
            # Matrix to compute error between UAV i and preceding UAV i-1
            error_matrix = self.uav_selection_matrices[i-1] - self.uav_selection_matrices[i]
            self.error_matrices.append(error_matrix)

        # Create position and velocity selection matrices (defined once during initialization)
        self.pos_selector = torch.zeros(2*self.dim, self.dim)
        self.pos_selector[:self.dim, :] = torch.eye(self.dim)
        
        self.vel_selector = torch.zeros(2*self.dim, self.dim)
        self.vel_selector[self.dim:, :] = torch.eye(self.dim)
        
        # Create control selection matrices for each follower dynamically
        self.control_selection_matrices = []
        for i in range(self.num_uavs - 1):  # For each follower
            control_selector = torch.zeros((self.num_uavs-1)*self.dim, self.dim)
            # Alternate between selecting position and velocity controls
            control_selector[(i % 2)*self.dim:(i % 2 + 1)*self.dim, :] = torch.eye(self.dim)
            self.control_selection_matrices.append(control_selector)

    def get_uav_state(self, x, uav_idx):
        """
        Extract state of a specific UAV from the combined state vector.
        
        Args:
            x (torch.Tensor): Combined state tensor [batch_size, num_uavs, state_dim]
            uav_idx (int): Index of UAV to extract
            
        Returns:
            torch.Tensor: State of specified UAV [batch_size, state_dim]
        """
        x_flat = x.reshape(x.shape[0], -1)  # Flatten to [batch_size, num_uavs*state_dim]
        selection_matrix = self.uav_selection_matrices[uav_idx]
        return x_flat @ selection_matrix  # [batch_size, state_dim]

    def error_state_transform(self, x):
        """
        Calculate error states between all follower UAVs and their preceding UAVs.
        
        Args:
            x (torch.Tensor): Combined state tensor [batch_size, num_uavs, state_dim]
            
        Returns:
            torch.Tensor: Combined error states for all followers [batch_size, (num_uavs-1)*state_dim]
        """
        batch_size = 1
        x_flat = x.reshape(batch_size, -1)  # Flatten to [batch_size, num_uavs*state_dim]
        
        error_states = []
        # Compute error for each follower UAV
        for i in range(1, self.num_uavs):
            # Get states of preceding and current UAVs
            preceding_state = x_flat @ self.uav_selection_matrices[i-1]
            following_state = x_flat @ self.uav_selection_matrices[i]
            
            # Extract position and velocity using matrix operations
            p_preceding = preceding_state @ self.pos_selector
            v_preceding = preceding_state @ self.vel_selector
            p_following = following_state @ self.pos_selector
            v_following = following_state @ self.vel_selector
            
            # Calculate desired position and errors
            p_desired = p_preceding - self.delta_ref
            p_error = p_desired + p_following*(-1)
            v_error = v_preceding + v_following*(-1)
            
            # Concatenate position and velocity errors
            error_state = torch.cat([p_error, v_error], dim=1)
            error_states.append(error_state)
        
        all_errors = torch.cat(error_states, dim=1)
        return all_errors

    def forward(self, x):
        """
        Forward pass through the combined network for 3 UAVs.
        
        Args:
            x (torch.Tensor): Combined state tensor [batch_size, num_uavs, state_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Current Lyapunov values,
                next states, and next Lyapunov values
        """
        batch_size = 1
        x = x.reshape(batch_size, -1)
        
        x_error = self.error_state_transform(x)
        current_V = self.V_net(x_error)

        
        control = self.controller(x_error)
        next_states = []
        # Apply dynamics to get next states
        for i in range(self.num_uavs):
            x_current_i = x@self.uav_selection_matrices[i]
            # Extract control signals using matrix multiplication
            if i == 0:
                control_i = control @ (self.control_selection_matrices[i]*0)
            else:
                control_i = control @ self.control_selection_matrices[i-1]
            next_state_i = self.system_dynamics(x_current_i, control_i)
            next_states.append(next_state_i)
        
        # Stack next states together
        next_state = torch.cat(next_states, dim=1)
        
        x_error_next = self.error_state_transform(next_state)
        next_V = self.V_net(x_error_next)
            
        return current_V, next_state, next_V

def combined_model(V_net, controllers, system_dynamics, output_file, state_dims, controlled_indices): 
    """
    Combine V_net, controllers and system dynamics into a single ONNX model for UAV formation
    
    Args:
        V_net: Vector Lyapunov network
        controllers: CombinedController with controllers for each follower UAV
        system_dynamics: System dynamics network
        output_file: Path to save combined ONNX model
        state_dims: List of state dimensions for each UAV
        controlled_indices: List of controlled UAV indices
    """
    
    # Create combined network
    combined_network = CombinedUAVNetwork(controllers, V_net, controlled_indices, state_dims, system_dynamics)
    
    # Save PyTorch model
    torch.save(combined_network, output_file.replace(".onnx", ".pth"))
    
    # Create example input for ONNX export
    num_uavs = len(state_dims)
    dim = state_dims[0] // 2
    
    # Create a reasonable formation state for the dummy input
    dummy_input = torch.zeros((1, num_uavs, state_dims[0]))
    
    # Set initial positions in a line formation
    for i in range(num_uavs):
        # Position: decreasing x-coordinate (10m spacing)
        dummy_input[0, i, 0] = -i * 10.0
        # All UAVs at same y, z
        dummy_input[0, i, 1] = 0.0
        dummy_input[0, i, 2] = 0.0
        
        # Velocity: all UAVs moving at same speed
        dummy_input[0, i, dim] = 5.0    # vx = 5 m/s
        dummy_input[0, i, dim+1] = 0.0  # vy = 0
        dummy_input[0, i, dim+2] = 0.0  # vz = 0

    dummy_input[0, 1, 3] = 10.0
    # Test the model
    #print("dummy_input", dummy_input)
    # output = combined_network(dummy_input)
    #print(output)
    
    # Export to ONNX
    torch.onnx.export(
        combined_network,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_states'],
        output_names=['lyapunov_values', 'next_states', 'next_lyapunov_values'],
    )
    
    # Simplify the ONNX model
    model = onnx.load(output_file)
    #model_simp, check = simplify(model)
    #assert check, "Simplified ONNX model could not be validated"
    #onnx.save(model_simp, output_file)

    
    print(f"Combined model exported to {output_file}")


if __name__ == "__main__":
    # Define dimensions for 3D UAV system
    dim = 3  # 3D space
    state_dim = 2 * dim  # Position and velocity for each dimension
    
    # Initialize networks with random weights (no loading required)
    V_net = VectorLyapunovNetwork(input_dim=state_dim, hidden_dim=64, G=None)
    combined_controller = CombinedController(input_dim=state_dim, output_dim=dim)

    combined_controller.controller_1.load_state_dict(torch.load("pre_train_model/controller_model.pth"))
    combined_controller.controller_2.load_state_dict(torch.load("pre_train_model/controller_model.pth"))
    system_dynamics = DynamicsNN(state_dim=state_dim, action_dim=dim)
    system_dynamics.load_state_dict(torch.load("pre_train_model/dynamics_model.pth"))
    
    # Generate the combined model
    output_file = "combined/combined_uav_model.onnx"
    combined_model(
        V_net=V_net,
        controllers=combined_controller,
        system_dynamics=system_dynamics,
        output_file=output_file,
        state_dims=[state_dim, state_dim, state_dim],  # 3 UAVs with same state dimension
        controlled_indices=[1, 2]  # Control follower UAVs
    )
    
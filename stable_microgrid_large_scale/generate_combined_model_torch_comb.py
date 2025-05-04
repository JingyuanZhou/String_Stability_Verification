import torch
import torch.nn as nn
import torch.onnx
from onnxsim import simplify
import onnx
import numpy as np
from networks import VectorLyapunovNetwork, ControllerNN, CombinedController, CombinedSystemDynamics

class CombinedMicrogridNetwork(nn.Module):
    def __init__(self, controller, V_net, controlled_indices, state_dims, system_dynamics):
        super(CombinedMicrogridNetwork, self).__init__()
        self.controller = controller
        self.V_net = V_net
        self.system_dynamics = system_dynamics
        self.controlled_indices = controlled_indices
        self.state_dims = state_dims
        self.num_inverters = len(state_dims)
        self.dt = 0.01
        self.batch_size = 1
        self.x_equilibrium = torch.zeros(self.batch_size, self.num_inverters * self.state_dims[0])
        for i in range(self.num_inverters):
            self.x_equilibrium[:, 3*i+1] = 2 * torch.pi * 50
        
        # Control bounds
        self.max_control = 1.0
        self.min_control = -1.0
        
        # Microgrid parameters
        self.omega_star = 2 * np.pi * 50  # Nominal frequency (50 Hz)
        

    def forward(self, x):
        """
        Forward pass through the combined network for microgrid inverters.
        
        Args:
            x (torch.Tensor): Combined state tensor [batch_size, num_inverters*state_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Current Lyapunov values,
                next states, and next Lyapunov values
        """
        
        # Calculate error states for VLF
        current_V = self.V_net(x)
        
        # Generate control inputs for controlled inverters
        control = self.controller(x)
        
        # Calculate next states
        x_true = x + self.x_equilibrium
        next_state = self.system_dynamics(x_true, control)
        next_state = next_state - self.x_equilibrium
        
        next_V = self.V_net(next_state)
            
        return current_V, next_state, next_V

def combined_model(V_net, controllers, system_dynamics, output_file, state_dims, controlled_indices, shared_lyapunov=False): 
    """
    Combine V_net, controllers and system dynamics into a single ONNX model for microgrid formation
    
    Args:
        V_net: Vector Lyapunov network
        controllers: CombinedController with controllers for each follower inverter
        system_dynamics: List of system dynamics networks for each inverter
        output_file: Path to save combined ONNX model
        state_dims: List of state dimensions for each inverter
        controlled_indices: List of controlled inverter indices
    """
    if not shared_lyapunov:
        for i in range(len(state_dims)):
            if i == 0:
                system_dynamics.dynamics[i].load_state_dict(torch.load(f"pre_train_model/dynamics_model_0.pth"))
                print(f"Loaded dynamics model for inverter {i}")
            elif i == len(state_dims) - 1:
                system_dynamics.dynamics[i].load_state_dict(torch.load(f"pre_train_model/dynamics_model_2.pth"))
                print(f"Loaded dynamics model for inverter {i}")
            else:
                system_dynamics.dynamics[i].load_state_dict(torch.load(f"pre_train_model/dynamics_model_1.pth"))
                print(f"Loaded dynamics model for inverter {i}")
    else:
        for i in range(3):
            if i == 0:
                system_dynamics.dynamics[i].load_state_dict(torch.load(f"pre_train_model/dynamics_model_0.pth"))
            elif i == 2:
                system_dynamics.dynamics[i].load_state_dict(torch.load(f"pre_train_model/dynamics_model_2.pth"))
            else:
                system_dynamics.dynamics[i].load_state_dict(torch.load(f"pre_train_model/dynamics_model_1.pth"))
    
    # Create combined network
    combined_network = CombinedMicrogridNetwork(controllers, V_net, controlled_indices, state_dims, system_dynamics)
    
    # Save PyTorch model
    torch.save(combined_network, output_file.replace(".onnx", ".pth"))
    
    # Create example input for ONNX export
    num_inverters = len(state_dims)
    state_dim = state_dims[0]  # Each inverter has the same state dimension (delta, omega, xi)
    
    # Create a reasonable microgrid state for the dummy input
    dummy_input = torch.zeros((1, num_inverters * state_dim))
    
    # Set initial states 
    for i in range(num_inverters):
        start_idx = i * state_dim
        # Initialize with reasonable values:
        # delta (phase angle offset from reference)
        dummy_input[0, start_idx] = 0.0 if i == 0 else 0.1 * i
        # omega (frequency)
        dummy_input[0, start_idx + 1] = 2 * np.pi * 50  # 50 Hz
        # xi (secondary controller state)
        dummy_input[0, start_idx + 2] = 0.0
    
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
    
    # Load and simplify the ONNX model
    model = onnx.load(output_file)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_file)
    
    print(f"Combined microgrid model exported to {output_file}")


if __name__ == "__main__":
    # Define dimensions for microgrid system
    state_dim = 3  # delta, omega, xi for each inverter
    num_inverters = 3  # Default to 3 inverters
    control_dim = 1  # Single control input per inverter
    
    # Initialize networks
    V_net = VectorLyapunovNetwork(input_dim=state_dim, hidden_dim=64, G=None)
    combined_controller = CombinedController(input_dim=state_dim, output_dim=control_dim)
    
    # Load dynamics models for each inverter
    system_dynamics = CombinedSystemDynamics(state_dim=state_dim, neighbor_dim=state_dim, control_dim=control_dim, hidden_dim=64)
    
    # Generate the combined model
    output_file = "combined/combined_microgrid_model.onnx"
    combined_model(
        V_net=V_net,
        controllers=combined_controller,
        system_dynamics=system_dynamics,
        output_file=output_file,
        state_dims=[state_dim] * num_inverters,  # Same state dimension for all inverters
        controlled_indices=list(range(1, num_inverters))  # Control all follower inverters
    )
    
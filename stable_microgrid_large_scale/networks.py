import torch
import torch.nn as nn
import numpy as np
from pre_train_model.learn_dynamics_control import DynamicsNN

class GraphCouplingMatrix(nn.Module):
    def __init__(self, N, G):
        """
        A learnable coupling matrix for microgrid communication graph.
        
        Args:
            N (int): Number of inverters.
            G (torch.Tensor): Initial adjacency matrix.
        """
        super(GraphCouplingMatrix, self).__init__()
        self.N = N
        self.G = G
        # Define a learnable parameter matrix
        self.coupling_matrix = torch.zeros(N, N)
        self.reset_parameters()
        self.coupling_matrix = nn.Parameter(self.coupling_matrix, requires_grad=True)
        
    def reset_parameters(self):
        """Initialize the coupling matrix with small values"""
        nn.init.uniform_(self.coupling_matrix, a=0.0001, b=0.001)
        # Zero out non-adjacent connections (for microgrid, usually a line topology)
        self.coupling_matrix = self.coupling_matrix * self.G
        
    def forward(self, G):
        """
        Forward pass to get the masked coupling matrix.
        
        Args:
            G (torch.Tensor): Adjacency matrix of shape (N, N), binary values {0,1}.
        
        Returns:
            torch.Tensor: Masked and nonnegative coupling matrix.
        """
        # Apply ReLU for nonnegativity
        A_tilde = torch.relu(self.coupling_matrix)
        
        # Apply adjacency matrix mask
        A_masked = torch.clamp(A_tilde * G, 0, 2)
        
        # Diagonal dominance adjustment
        row_sum = torch.sum(A_masked, dim=1) - 2*torch.diag(A_masked)  # Sum of non-diagonal elements
        diag_values = row_sum + 1e-5  # Make diagonal elements slightly larger than row sum
        A_diag = torch.diag_embed(diag_values)  # Create diagonal matrix
        
        # Update diagonal elements to ensure diagonal dominance
        A_final = A_masked + A_diag  
        
        return A_final

class VectorLyapunovNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, G=None, num_inverters=3, shared_lyapunov=False):
        """
        Vector Lyapunov Function Network for microgrid stability.
        
        Args:
            input_dim (int): Dimension of state (δ, ω, ξ)
            hidden_dim (int): Dimension of hidden layers
            G (torch.Tensor): Communication graph adjacency matrix
        """
        super(VectorLyapunovNetwork, self).__init__()
        self.shared_lyapunov = shared_lyapunov
        self.num_inverters = num_inverters

        # Network that maps states to scalar Lyapunov values
        self.networks = nn.ModuleList()
        
        if not shared_lyapunov:
            # Dynamically create neural networks for each inverter
            for i in range(num_inverters):
                # Determine input dimension based on connections
                # Middle inverters have connections to two neighbors
                if 0 < i < num_inverters - 1:
                    # Middle inverters connect to both neighbors (2 extra inputs)
                    net_input_dim = input_dim + 2
                else:
                    # Edge inverters connect to only one neighbor
                    net_input_dim = input_dim
                    
                network = nn.Sequential(
                    nn.Linear(net_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                self.networks.append(network)
        else:
            # Shared Lyapunov
            for i in range(2):
                if i==0:
                    net_input_dim = input_dim 
                else:
                    net_input_dim = input_dim + 2
                network = nn.Sequential(
                    nn.Linear(net_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                self.networks.append(network)
        if G is not None:
            self.coupling_matrix = GraphCouplingMatrix(self.num_inverters, G)


    def forward(self, state):
        """
        Compute Lyapunov function value for given state.
        
        Args:
            state (torch.Tensor): State tensor [batch_size, num_inverters*3]
                                Contains phase angles, frequencies, and controller states
        
        Returns:
            torch.Tensor: Lyapunov function value [batch_size, 2]
        """
        # error state -> e_delta_ij = e_delta_i - e_delta_j, e_omega_i = e_omega_i - e_omega_star, e_xi_ij = e_xi_i - e_xi_j
        V = []
        for i in range(self.num_inverters):
            e_delta_ij = []
            e_omega_i = state[:, i*3+1:i*3+2]
            e_xi_i = []
            if i-1 >= 0:
                e_delta_ij.append((state[:, i*3:i*3+1] - state[:, (i-1)*3:(i-1)*3+1]))
                e_xi_i.append((state[:, i*3+2:i*3+3] - state[:, (i-1)*3+2:(i-1)*3+3]))
            if i+1 < self.num_inverters:
                e_delta_ij.append((state[:, i*3:i*3+1] - state[:, (i+1)*3:(i+1)*3+1]))
                e_xi_i.append((state[:, i*3+2:i*3+3] - state[:, (i+1)*3+2:(i+1)*3+3]))
            e_delta_ij = torch.cat(e_delta_ij, dim=-1)
            e_xi_i = torch.cat(e_xi_i, dim=-1)  
            #e_omega_i = e_omega_i.unsqueeze(1)
            #print(e_delta_ij.shape, e_omega_i.shape, e_xi_i.shape)
            error_state = torch.cat([e_delta_ij, e_omega_i, e_xi_i], dim=-1)
            equilibrium_state = torch.zeros_like(error_state)
            if not self.shared_lyapunov:
                V_i = self.networks[i](error_state) - self.networks[i](equilibrium_state) + 0.001
            else:
                if i<1:
                    V_i = self.networks[0](error_state) - self.networks[0](equilibrium_state) + 0.001
                elif i>1 and i<self.num_inverters-1:
                    V_i = self.networks[1](error_state) - self.networks[1](equilibrium_state) + 0.001
                elif i==self.num_inverters-1:
                    V_i = self.networks[0](error_state) - self.networks[0](equilibrium_state) + 0.001
            V.append(V_i)
        V = torch.cat(V, dim=-1)
        return V

class ControllerNN(nn.Module):
    """Neural network for learning microgrid controller"""
    def __init__(self, input_dim, output_dim=1, hidden_dim=64):
        """
        Controller network for microgrid.
        
        Args:
            input_dim (int): Input dimension (state + neighbor states + target state)
            output_dim (int): Output dimension (control input)
            hidden_dim (int): Hidden layer dimension
        """
        super(ControllerNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): Input state [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Control input [batch_size, output_dim]
        """
        return self.net(state)

class CombinedController(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, num_inverters=3, shared_lyapunov=False):
        """
        Combined controller for multiple inverters.
        
        Args:
            input_dim (int): Input dimension per inverter
            output_dim (int): Output dimension per inverter
            hidden_dim (int): Hidden layer dimension
        """
        super(CombinedController, self).__init__()
        self.num_inverters = num_inverters
        self.controllers = nn.ModuleList()
        self.shared_lyapunov = shared_lyapunov
        if not shared_lyapunov:
            for i in range(num_inverters):
                if i == 0:
                    self.controllers.append(ControllerNN(input_dim + input_dim*1, output_dim, hidden_dim))
                elif i == num_inverters - 1:
                    self.controllers.append(ControllerNN(input_dim + input_dim*1, output_dim, hidden_dim))
                else:
                    self.controllers.append(ControllerNN(input_dim + input_dim*2, output_dim, hidden_dim))
        else:
            for i in range(3):
                if i == 0 or i == 2:
                    self.controllers.append(ControllerNN(input_dim + input_dim*1, output_dim, hidden_dim))
                else:
                    self.controllers.append(ControllerNN(input_dim + input_dim*2, output_dim, hidden_dim))
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): Input state [batch_size, num_inverters*input_dim]
        
        Returns:
            torch.Tensor: Control inputs [batch_size, num_inverters*output_dim]
        """
        # Split state for each inverter
        inverter_states = []
        state_dim = 3  # Assuming each inverter state has dimension 3
        
        for i in range(self.num_inverters):
            inverter_states.append(state[:, i*state_dim:(i+1)*state_dim])
        
        # Create combined states based on neighbor connections
        combined_states = []
        control_outputs = []
        
        for i in range(self.num_inverters):
            # Start with own state
            combined_state = [inverter_states[i]]
            
            # Add left neighbor if it exists
            if i > 0:
                combined_state.append(inverter_states[i-1])
                
            # Add right neighbor if it exists
            if i < self.num_inverters - 1:
                combined_state.append(inverter_states[i+1])
                
            # Concatenate to form the complete combined state
            combined_state = torch.cat(combined_state, dim=-1)
            
            # Apply the controller
            if not self.shared_lyapunov:
                control_output = self.controllers[i](combined_state)
            else:
                if i<=1:
                    control_output = self.controllers[i](combined_state)
                elif i>1 and i<self.num_inverters-1:
                    control_output = self.controllers[1](combined_state)
                elif i==self.num_inverters-1:
                    control_output = self.controllers[2](combined_state)
            control_outputs.append(control_output)
        
        # Concatenate all control outputs
        return torch.cat(control_outputs, dim=-1)
    

class CombinedSystemDynamics(nn.Module):
    def __init__(self, state_dim, neighbor_dim, control_dim, hidden_dim=64, num_inverters=3, shared_lyapunov=False):
        super(CombinedSystemDynamics, self).__init__()
        self.num_inverters = num_inverters
        self.dynamics = nn.ModuleList()
        self.shared_lyapunov = shared_lyapunov
        if not shared_lyapunov:
            for i in range(num_inverters):
                if i == 0 or i == num_inverters - 1:
                    self.dynamics.append(DynamicsNN(state_dim, neighbor_dim, control_dim, hidden_dim))
                else:
                    self.dynamics.append(DynamicsNN(state_dim, neighbor_dim*2, control_dim, hidden_dim))
        else:
            for i in range(3):
                if i == 0 or i == 2:
                    self.dynamics.append(DynamicsNN(state_dim, neighbor_dim, control_dim, hidden_dim))
                else:
                    self.dynamics.append(DynamicsNN(state_dim, neighbor_dim*2, control_dim, hidden_dim))

        self.control_selection_list = []
        for i in range(num_inverters):
            self.control_selection_list.append(torch.zeros(num_inverters, 1))
            self.control_selection_list[i][i, 0] = 1


    def forward(self, state, control):
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): Input state [batch_size, num_inverters*input_dim]
        
        Returns:
            torch.Tensor: Control inputs [batch_size, num_inverters*output_dim]
        """
        state_ego_list = []
        control_list = []
        next_state_list = []
        for i in range(self.num_inverters):
            state_ego_i = state[:, i*3:(i+1)*3]  # [state(3) + neighbors(6) + target(3)]
            state_ego_list.append(state_ego_i)
            control_i = control@self.control_selection_list[i]#.squeeze(1)
            control_list.append(control_i)

        for i in range(self.num_inverters):
            state_i_neighbor = []
            if i-1 >= 0:
                state_i_neighbor.append(state_ego_list[i-1])
            if i+1 < self.num_inverters:
                state_i_neighbor.append(state_ego_list[i+1])
            state_i_neighbor = torch.cat(state_i_neighbor, dim=-1)
            
            if not self.shared_lyapunov:
                next_state_i = self.dynamics[i](state_ego_i, state_i_neighbor, control_i)
            else:
                if i<=1:
                    next_state_i = self.dynamics[i](state_ego_i, state_i_neighbor, control_i)
                elif i>1 and i<self.num_inverters-1:
                    next_state_i = self.dynamics[1](state_ego_i, state_i_neighbor, control_i)
                elif i==self.num_inverters-1:
                    next_state_i = self.dynamics[2](state_ego_i, state_i_neighbor, control_i)
            next_state_list.append(next_state_i)

        next_state = torch.cat(next_state_list, dim=-1)
        return next_state
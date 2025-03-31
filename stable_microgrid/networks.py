import torch
import torch.nn as nn
import numpy as np

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
        nn.init.uniform_(self.coupling_matrix, a=0.01, b=0.1)
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
    def __init__(self, input_dim=3, hidden_dim=64, G=None):
        """
        Vector Lyapunov Function Network for microgrid stability.
        
        Args:
            input_dim (int): Dimension of state (δ, ω, ξ)
            hidden_dim (int): Dimension of hidden layers
            G (torch.Tensor): Communication graph adjacency matrix
        """
        super(VectorLyapunovNetwork, self).__init__()
        
        self.num_inverters = 3

        # Network that maps states to scalar Lyapunov values
        self.network_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.network_2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.network_3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        if G is not None:
            self.coupling_matrix = GraphCouplingMatrix(self.num_inverters, G)

        # matrix for inverter state selection [batch_size, n_inverters*3] -> [batch_size, 3]
        self.state_inv1_matrix = torch.zeros((self.num_inverters)*3, 3)
        self.state_inv1_matrix[:3, :] = torch.eye(3)
        self.state_inv2_matrix = torch.zeros((self.num_inverters)*3, 3)
        self.state_inv2_matrix[3:6, :] = torch.eye(3)
        self.state_inv3_matrix = torch.zeros((self.num_inverters)*3, 3)
        self.state_inv3_matrix[6:9, :] = torch.eye(3)


    def forward(self, state):
        """
        Compute Lyapunov function value for given state.
        
        Args:
            state (torch.Tensor): State tensor [batch_size, num_inverters*3]
                                Contains phase angles, frequencies, and controller states
        
        Returns:
            torch.Tensor: Lyapunov function value [batch_size, 2]
        """
        state_inv1 = state @ self.state_inv1_matrix
        state_inv2 = state @ self.state_inv2_matrix
        state_inv3 = state @ self.state_inv3_matrix

        equilibrium_state = state_inv1*0

        V_1 = self.network_1(state_inv1) + self.network_1(equilibrium_state)*(-1) + 0.001
        V_2 = self.network_2(state_inv2) + self.network_2(equilibrium_state)*(-1) + 0.001
        V_3 = self.network_3(state_inv3) + self.network_3(equilibrium_state)*(-1) + 0.001
        V = torch.cat([V_1, V_2, V_3], dim=-1)
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
    def __init__(self, input_dim, output_dim=1, hidden_dim=64):
        """
        Combined controller for multiple inverters.
        
        Args:
            input_dim (int): Input dimension per inverter
            output_dim (int): Output dimension per inverter
            hidden_dim (int): Hidden layer dimension
        """
        super(CombinedController, self).__init__()
        self.num_inverters = 3
        self.controller_1 = ControllerNN(input_dim + input_dim*1, output_dim, hidden_dim)
        self.controller_2 = ControllerNN(input_dim + input_dim*2, output_dim, hidden_dim)
        self.controller_3 = ControllerNN(input_dim + input_dim*1, output_dim, hidden_dim)
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): Input state [batch_size, num_inverters*input_dim]
        
        Returns:
            torch.Tensor: Control inputs [batch_size, num_inverters*output_dim]
        """
        # Split state for each inverter
        state_ego_1 = state[:, :3]  # [state(3) + neighbor(3) + target(3)]
        state_ego_2 = state[:, 3:6]  # [state(3) + neighbors(6) + target(3)]
        state_ego_3 = state[:, 6:9]  # [state(3) + neighbor(3) + target(3)]
        
        state_1 = torch.cat([state_ego_1, state_ego_2], dim=-1)
        state_2 = torch.cat([state_ego_2, state_ego_1, state_ego_3], dim=-1)
        state_3 = torch.cat([state_ego_3, state_ego_1], dim=-1)

        # Get control inputs
        control_1 = self.controller_1(state_1)
        control_2 = self.controller_2(state_2)
        control_3 = self.controller_3(state_3)
        
        return torch.cat([control_1, control_2, control_3], dim=-1) 
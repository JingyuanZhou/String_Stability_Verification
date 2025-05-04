import torch
import torch.nn as nn
import numpy as np

class GraphCouplingMatrix(nn.Module):
    def __init__(self, N, G):
        """
        A learnable coupling matrix.
        
        Args:
            N (int): Number of vehicles.
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
        # Zero out non-adjacent connections (for UAV formation, usually a line topology)
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
    def __init__(self, input_dim=6, hidden_dim=96, G = None):
        """
        Vector Lyapunov Function Network for UAV formation control.
        
        Args:
            input_dim (int): Dimension of error state (position error + velocity error)
            hidden_dim (int): Dimension of hidden layers
        """
        super(VectorLyapunovNetwork, self).__init__()
        
        self.num_UAVs = 3


        # Network that maps error states to scalar Lyapunov values
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

        #self.equilibrium_state = torch.zeros_like(error_state)

        if G is not None:
            self.coupling_matrix = GraphCouplingMatrix(self.num_UAVs, G)

        # matrix for UAV state selection [batch_size, n_UAVs*6] -> [batch_size, 6]
        self.state_UAV1_matrix = torch.zeros((self.num_UAVs-1)*6, 6)
        self.state_UAV1_matrix[:6, :] = torch.eye(6)
        self.state_UAV2_matrix = torch.zeros((self.num_UAVs-1)*6, 6)
        self.state_UAV2_matrix[6:12, :] = torch.eye(6)

        self.mask = torch.zeros(12,12)
        self.mask[0,0] = 1
        self.mask[3,3] = 1
        self.mask[6,6] = 1
        self.mask[9,9] = 1

    def forward(self, error_state):
        """
        Compute Lyapunov function value for given error state.
        
        Args:
            error_state (torch.Tensor): Error state tensor [batch_size, input_dim]
                                         Contains position and velocity errors
        
        Returns:
            torch.Tensor: Lyapunov function value [batch_size, 1]
        """
        # Compute Lyapunov value (must be positive for non-zero states)
        # set y,z,vy,vz to 0

        #error_state = error_state @ self.mask
        error_state_UAV1 = error_state @ self.state_UAV1_matrix
        error_state_UAV2 = error_state @ self.state_UAV2_matrix

        equilibrium_state = error_state_UAV1*0

        V_1 = self.network_1(error_state_UAV1) + self.network_1(equilibrium_state)*(-1) + 0.001
        V_2 = self.network_2(error_state_UAV2) + self.network_2(equilibrium_state)*(-1) + 0.001
        V = torch.cat([V_1, V_2], dim=-1)
        return V

class ControllerNN(nn.Module):
    """Neural network for learning UAV controller"""
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=64):
        super(ControllerNN, self).__init__()
        # Input: position error (3D) and velocity error (3D)
        # Output: control actions (3D)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim
        
        self.mask = torch.zeros(output_dim, output_dim)
        self.mask[0,0] = 1
    def forward(self, error_state):
        raw_output = self.net(error_state)
        # only use the first output_dim elements
        return raw_output
    
class CombinedController(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=64):
        super(CombinedController, self).__init__()
        self.num_UAVs = 3
        self.controller_1 = ControllerNN(input_dim, output_dim, hidden_dim)
        self.controller_2 = ControllerNN(input_dim, output_dim, hidden_dim)
        self.state_UAV1_matrix = torch.zeros((self.num_UAVs-1)*6, 6)
        self.state_UAV1_matrix[:6, :] = torch.eye(6)
        self.state_UAV2_matrix = torch.zeros((self.num_UAVs-1)*6, 6)
        self.state_UAV2_matrix[6:12, :] = torch.eye(6)
        
    def forward(self, error_state):
        error_state_UAV1 = error_state @ self.state_UAV1_matrix
        error_state_UAV2 = error_state @ self.state_UAV2_matrix
        control_1 = self.controller_1(error_state_UAV1)
        control_2 = self.controller_2(error_state_UAV2)
        return torch.cat([control_1, control_2], dim=-1)
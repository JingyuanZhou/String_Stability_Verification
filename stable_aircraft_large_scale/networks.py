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
    def __init__(self, input_dim=6, hidden_dim=96, G=None, num_UAVs=3, shared_Lyapunov=True):
        """
        Vector Lyapunov Function Network for UAV formation control.
        
        Args:
            input_dim (int): Dimension of error state (position error + velocity error)
            hidden_dim (int): Dimension of hidden layers
            G (torch.Tensor): Adjacency matrix
            num_UAVs (int): Number of UAVs in the formation
        """
        super(VectorLyapunovNetwork, self).__init__()
        
        self.num_UAVs = num_UAVs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.shared_Lyapunov = shared_Lyapunov

        # Dynamically create networks for each UAV (except leader)
        self.networks = nn.ModuleList()
        if self.shared_Lyapunov:
            for i in range(3):
                network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                self.networks.append(network)
        else:
            for i in range(self.num_UAVs - 1):
                network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                self.networks.append(network)

        if G is not None:
            self.coupling_matrix = GraphCouplingMatrix(self.num_UAVs, G)

        # Dynamically create state selection matrices
        self.state_selection_matrices = []
        for i in range(self.num_UAVs - 1):
            matrix = torch.zeros((self.num_UAVs-1)*6, 6)
            matrix[i*6:(i+1)*6, :] = torch.eye(6)
            self.state_selection_matrices.append(matrix)

        self.mask = torch.zeros((self.num_UAVs-1)*6, (self.num_UAVs-1)*6)
        for i in range(self.num_UAVs-1):
            self.mask[i*3, i*3] = 1
            self.mask[i*3+3, i*3+3] = 1

    def forward(self, error_state):
        """
        Compute Lyapunov function value for given error state.
        
        Args:
            error_state (torch.Tensor): Error state tensor [batch_size, input_dim]
                                         Contains position and velocity errors
        
        Returns:
            torch.Tensor: Lyapunov function values [batch_size, num_UAVs-1]
        """
        # Create a zero state as equilibrium reference
        
        # Compute Lyapunov values for each UAV
        V_values = []
        for i in range(self.num_UAVs - 1):
            if self.shared_Lyapunov:
                error_state_UAV = error_state @ self.state_selection_matrices[i]
                equilibrium_state = error_state_UAV * 0
                if i<=2:
                    V_i = self.networks[i](error_state_UAV) + self.networks[i](equilibrium_state)*(-1) + 0.001
                V_values.append(V_i)
            else:
                error_state_UAV = error_state @ self.state_selection_matrices[i]
                equilibrium_state = error_state_UAV * 0
                V_i = self.networks[i](error_state_UAV) + self.networks[i](equilibrium_state)*(-1) + 0.001
                V_values.append(V_i)
        
        # Concatenate all Lyapunov values
        V = torch.cat(V_values, dim=-1)
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
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=64, num_UAVs=3, shared_Lyapunov=True):
        super(CombinedController, self).__init__()
        self.num_UAVs = num_UAVs
        self.shared_Lyapunov = shared_Lyapunov
        
        # Dynamically create controllers for each UAV (except leader)
        self.controllers = nn.ModuleList()
        if self.shared_Lyapunov:    
            for i in range(3):
                controller = ControllerNN(input_dim, output_dim, hidden_dim)
                self.controllers.append(controller)
        else:
            for i in range(self.num_UAVs - 1):
                controller = ControllerNN(input_dim, output_dim, hidden_dim)
                self.controllers.append(controller)
        
        # Dynamically create state selection matrices
        self.state_selection_matrices = []
        for i in range(self.num_UAVs - 1):
            matrix = torch.zeros((self.num_UAVs-1)*6, 6)
            matrix[i*6:(i+1)*6, :] = torch.eye(6)
            self.state_selection_matrices.append(matrix)
        
    def forward(self, error_state):
        control_outputs = []
        for i in range(self.num_UAVs - 1):
            if self.shared_Lyapunov:
                if i<=2:
                    error_state_UAV = error_state @ self.state_selection_matrices[i]
                    control_i = self.controllers[i](error_state_UAV)
            else:
                error_state_UAV = error_state @ self.state_selection_matrices[i]
                control_i = self.controllers[i](error_state_UAV)
            control_outputs.append(control_i)
        
        return torch.cat(control_outputs, dim=-1)
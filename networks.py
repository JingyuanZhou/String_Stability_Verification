import torch
import torch.nn as nn
import numpy as np

class GraphCouplingMatrix(nn.Module):
    def __init__(self, N):
        """
        A learnable coupling matrix.
        
        Args:
            N (int): Number of vehicles.
        """
        super(GraphCouplingMatrix, self).__init__()
        self.N = N
        # 直接定义一个可学习的参数矩阵
        self.coupling_matrix = nn.Parameter(torch.zeros(N, N), requires_grad=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the coupling matrix with small values"""
        nn.init.uniform_(self.coupling_matrix, a = 0.045, b = 0.048)# 0.01
        
    def forward(self, G):
        """
        Forward pass to get the masked coupling matrix.
        
        Args:
            G (torch.Tensor): Adjacency matrix of shape (N, N), binary values {0,1}.
        
        Returns:
            torch.Tensor: Masked and nonnegative coupling matrix of shape (N, N).
        """
        # Apply ReLU for nonnegativity
        A_tilde = torch.relu(self.coupling_matrix)
        
        # Apply adjacency matrix mask
        A_masked = A_tilde * G
        
        return A_masked

class VectorLyapunovNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(VectorLyapunovNetwork, self).__init__()

        self.num_vehicles = len(state_dim)
        self.one_state_dim = state_dim[0]
        self.all_state_dim = sum(state_dim)

        # 为4个跟随车辆创建网络
        self.network_1 = nn.Sequential(
            nn.Linear(self.one_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.network_2 = nn.Sequential(
            nn.Linear(self.one_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.network_3 = nn.Sequential(
            nn.Linear(self.one_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.network_4 = nn.Sequential(
            nn.Linear(self.one_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.coupling_matrix = GraphCouplingMatrix(self.num_vehicles)

        # 创建选择矩阵
        W1 = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        W2 = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        W3 = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        W4 = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        W_star = torch.zeros(self.all_state_dim, self.one_state_dim, requires_grad=False)
        
        # 设置选择矩阵的元素
        W1[self.one_state_dim:2*self.one_state_dim, :] = torch.eye(self.one_state_dim)  # 第2辆车
        W2[2*self.one_state_dim:3*self.one_state_dim, :] = torch.eye(self.one_state_dim)  # 第3辆车
        W3[3*self.one_state_dim:4*self.one_state_dim, :] = torch.eye(self.one_state_dim)  # 第4辆车
        W4[4*self.one_state_dim:5*self.one_state_dim, :] = torch.eye(self.one_state_dim)  # 第5辆车
        W_star[0:self.one_state_dim, :] = torch.eye(self.one_state_dim)  # 参考状态

        self.register_buffer('W1', W1)
        self.register_buffer('W2', W2)
        self.register_buffer('W3', W3)
        self.register_buffer('W4', W4)
        self.register_buffer('W_star', W_star)

    def forward(self, x, x_star):
        """
        计算 Lyapunov 函数值

        参数:
        - x (Tensor): 输入张量，形状为 [batch_size, all_state_dim]
        - x_star (Tensor): 参考状态张量，形状为 [batch_size, num_star, one_state_dim]

        返回:
        - V (Tensor): Lyapunov 函数值，形状为 [batch_size, 4]
        """
        x = x.view(-1, self.all_state_dim)
        x_star = x_star.view(-1, self.all_state_dim)

        # 提取每辆车的状态
        x1 = torch.matmul(x, self.W1)  # 第2辆车(CAV)
        x2 = torch.matmul(x, self.W2)  # 第3辆车(HDV)
        x3 = torch.matmul(x, self.W3)  # 第4辆车(CAV)
        x4 = torch.matmul(x, self.W4)  # 第5辆车(HDV)
        x_star_1 = torch.matmul(x_star, self.W_star)

        # 计算每辆车的Lyapunov函数值
        V_1 = self.network_1(x1) - self.network_1(x_star_1) + 0.001  # CAV
        V_2 = self.network_2(x2) - self.network_2(x_star_1) + 0.001  # HDV
        V_3 = self.network_3(x3) - self.network_3(x_star_1) + 0.001  # CAV
        V_4 = self.network_4(x4) - self.network_4(x_star_1) + 0.001  # HDV

        # 组合所有Lyapunov函数值
        V = torch.cat([V_1, V_2, V_3, V_4], dim=1)
            
        return V
    

class NetworkController(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=30):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, control_dim)
        )
        self.state_dim = state_dim
        self.W_change_state_position = torch.zeros(self.state_dim, self.state_dim, requires_grad=False)
        #index from 0 1 2 3 4 5 to 1 3 5 0 2 4
        self.W_change_state_position[1, 0] = 1
        self.W_change_state_position[3, 1] = 1
        self.W_change_state_position[5, 2] = 1
        self.W_change_state_position[0, 3] = 1
        self.W_change_state_position[2, 4] = 1
        self.W_change_state_position[4, 5] = 1
        
    def forward(self, x, x_star, u_star, u_bounds):
        """
        Compute control input with clamping
        """
        u_min, u_max = u_bounds
        x = x.reshape(-1, self.state_dim)
        x = x @ self.W_change_state_position
        x_star = x_star.reshape(-1, self.state_dim)
        x_star = x_star @ self.W_change_state_position
        phi_pi = self.network(x)
        phi_pi_star = self.network(x_star)
        u = phi_pi- phi_pi_star #torch.clamp(phi_pi, u_min, u_max)# + u_star
        return u
    
class CombinedControllers(nn.Module):
    def __init__(self, controllers):
        super().__init__()
        self.controllers = controllers
        CAV_indices = [1, 3]
        self.CAV_controller_1 = controllers[CAV_indices[0]]
        self.CAV_controller_2 = controllers[CAV_indices[1]]

    def forward(self, x, x_star, u_star, u_bounds):
        """
        Compute control inputs for all vehicles
        """
        u_1 = self.CAV_controller_1(x, x_star, u_star, u_bounds)
        u_2 = self.CAV_controller_2(x, x_star, u_star, u_bounds)
        return torch.cat([u_1, u_2], dim=1)

class system_network(nn.Module):
    def __init__(self, state_dim, hidden_dim=30):
        super().__init__()
        self.state_dim = state_dim

        # network with state and control input as input
        self.network = nn.Sequential(
            nn.Linear(state_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Compute next state
        """
        x = x.reshape(-1, self.state_dim)
        a = self.network(x)
        #a = torch.clip(a, -5.0, 5.0)
        return a
    
def orthogonal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        nn.init.constant_(layer.bias, 0)
    return layer

def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    # 对每一层进行初始化
    trunk = nn.Sequential(*mods)
    for m in trunk.modules():
        orthogonal_init(m)
    
    return trunk

class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim = 30, hidden_depth = 2):
        super().__init__()
        self.Q = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

    def forward(self, obs, action, both=False):
        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)
        return q

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=30, hidden_depth = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)


    def forward(self, obs, action, both=False):
        obs = obs.reshape(-1, self.obs_dim)
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if both:
            return q1, q2
        return torch.min(q1, q2)
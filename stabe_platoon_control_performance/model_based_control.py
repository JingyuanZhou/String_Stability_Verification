"""
编队控制的模型基控制器 (Model-based Controllers for Platoon Control).

提供两种控制器，接口与 NetworkController 一致 (forward(x, x_star, u_star, u_bounds) -> u):

1) LCC (Linear Cruise Controller): 线性反馈
    u = K^T (x - x*)，对重排后全状态（自车+后车+前车）做增益反馈，经 u_bounds 限幅。

2) MPC (Model Predictive Control): 基于 CAV 离散动力学的前向预测，每步求解有限时域 QP，
    取首步控制 u_0 作为输出；动力学 s'=s+dt*(v_lead-v), v'=v+dt*u，v_lead 在时域内取常值。
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize


# 与 NetworkController 相同的状态重排矩阵：将全局状态 [v0_s, v0_v, v1_s, v1_v, ...] 
# 重排为 [v1_s, v1_v, v2_s, v2_v, ..., v0_s, v0_v]，使 CAV（索引 1）的局部状态在前 2 维
def _make_state_reorder_matrix(state_dim: int, device):
    """state_dim = num_vehicles * 2，当前为 10（5 车）."""
    W = torch.zeros(state_dim, state_dim, requires_grad=False, device=device)
    # 映射: 0,1,2,3,4 -> 5,0,6,1,7,2,8,3,9,4 (即 1 3 5 7 9 0 2 4 6 8)
    W[1, 0] = 1
    W[3, 1] = 1
    W[5, 2] = 1
    W[7, 3] = 1
    W[9, 4] = 1
    W[0, 5] = 1
    W[2, 6] = 1
    W[4, 7] = 1
    W[6, 8] = 1
    W[8, 9] = 1
    return W


class LinearFeedbackController(nn.Module):
    """
    编队中 CAV 的线性反馈控制器：考虑自车与后车状态，各有独立增益。
    重排后状态顺序为 [自车_s, 自车_v, 后车2_s, 后车2_v, 后车3_s, 后车3_v, 后车4_s, 后车4_v, 前车_s, 前车_v]。
    输入为全局状态 x、期望状态 x_star、期望控制 u_star 与控制限幅 u_bounds；
    输出为标量加速度 u，与 NetworkController 接口一致。
    """

    def __init__(
        self,
        state_dim: int = 10,
        control_dim: int = 1,
        k_ego_spacing: float = 0.8,
        k_ego_velocity: float = -0.6,
        k_following_spacing: float = -0.01,
        k_following_velocity: float = -0.01,
        k_leader_spacing: float = 0.0,
        k_leader_velocity: float = -0.4,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.device = device
        # 重排后: [自车_s, 自车_v, 后车2_s, 后车2_v, 后车3_s, 后车3_v, 后车4_s, 后车4_v, 前车_s, 前车_v]
        # 自车 2 维 + 后车 3 辆×2 维 + 前车 2 维 = 10
        K_list = [
            k_ego_spacing, k_ego_velocity,
            k_following_spacing, k_following_velocity,
            k_following_spacing, k_following_velocity,
            k_following_spacing, k_following_velocity,
            k_leader_spacing, k_leader_velocity,
        ]
        self.register_buffer("K", torch.tensor(K_list, dtype=torch.float32, device=device))
        self.register_buffer(
            "W_change_state_position",
            _make_state_reorder_matrix(state_dim, device),
        )

    def forward(self, x, x_star, u_star, u_bounds):
        """
        计算线性反馈控制量并限幅。u = u* - K^T (x - x*)，对重排后全状态做反馈。

        Args:
            x: 全局状态，shape (batch_size, state_dim) 或 (batch_size, num_vehicles, 2)
            x_star: 期望状态，shape 与 x 一致
            u_star: 期望控制，shape (batch_size,) 或 (batch_size, 1)
            u_bounds: (u_min, u_max) 标量或与 batch 兼容的 tensor

        Returns:
            u: 控制量（加速度），shape (batch_size, 1)
        """
        u_min, u_max = u_bounds
        u_min = u_min if isinstance(u_min, (int, float)) else u_min.to(self.device)
        u_max = u_max if isinstance(u_max, (int, float)) else u_max.to(self.device)
        x = x.reshape(-1, self.state_dim).to(self.device)
        x_star = x_star.reshape(-1, self.state_dim).to(self.device)
        x = x @ self.W_change_state_position
        x_star = x_star @ self.W_change_state_position
        e = x - x_star
        u_star = u_star.reshape(-1, 1).to(self.device)
        u = (e * self.K.unsqueeze(0)).sum(dim=1, keepdim=True)
        u = torch.clamp(u, u_min, u_max)
        return u


def get_linear_feedback_controller(
    state_dim: int = 10,
    control_dim: int = 1,
    k_ego_spacing: float = 0.1,
    k_ego_velocity: float = -0.3,
    k_following_spacing: float = -0.01,
    k_following_velocity: float = -0.05,
    k_leader_spacing: float = 0.0,
    k_leader_velocity: float = -0.8,
    device: str = None,
) -> LinearFeedbackController:
    """构造 LCC（线性反馈）控制器实例。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return LinearFeedbackController(
        state_dim=state_dim,
        control_dim=control_dim,
        k_ego_spacing=k_ego_spacing,
        k_ego_velocity=k_ego_velocity,
        k_following_spacing=k_following_spacing,
        k_following_velocity=k_following_velocity,
        k_leader_spacing=k_leader_spacing,
        k_leader_velocity=k_leader_velocity,
        device=device,
    )


# --------------- MPC ---------------


class MPCController(nn.Module):
    """
    编队中 CAV 的模型预测控制器（MPC）。
    基于 CAV 的离散动力学在有限时域上求解二次型代价的最小化，取首步控制 u_0 应用。
    动力学（前车速度在时域内取常值）:
        s_{k+1} = s_k + dt * (v_lead - v_k)
        v_{k+1} = v_k + dt * u_k
    代价: sum_{k=0}^{N-1} [ Q_s*(s_k-s*)^2 + Q_v*(v_k-v*)^2 + Q_vel_lead*(v_k-v_lead)^2
        + Q_s_lead*(s_k-s_lead)^2 + R*(u_k-u*)^2 ] + 初始时刻对后车的惩罚 + 终端代价。
    重排后状态: [自车_s,v, 后车2_s,v, 后车3_s,v, 后车4_s,v, 前车_s,v]。
    接口与 NetworkController 一致。
    """

    def __init__(
        self,
        state_dim: int = 10,
        control_dim: int = 1,
        dt: float = 0.1,
        horizon: int = 10,
        Q_spacing: float = 0.3,
        Q_velocity: float = 0.5,
        R_control: float = 0.01,
        Q_vel_lead: float = 0.5,
        Q_s_lead: float = 0.0,
        Q_following_s: float = 0.01,
        Q_following_v: float = 0.02,
        Q_terminal_spacing: float = 2.0,
        Q_terminal_velocity: float = 2.0,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.device = device
        self.dt = dt
        self.horizon = int(horizon)
        self.Q_s = Q_spacing
        self.Q_v = Q_velocity
        self.R = R_control
        self.Q_vel_lead = Q_vel_lead
        self.Q_s_lead = Q_s_lead
        self.Q_following_s = Q_following_s
        self.Q_following_v = Q_following_v
        self.Q_s_term = Q_terminal_spacing
        self.Q_v_term = Q_terminal_velocity
        self.register_buffer(
            "W_change_state_position",
            _make_state_reorder_matrix(state_dim, device),
        )

    def _rollout_and_cost(self, x_full: np.ndarray, x_star_full: np.ndarray, u_star: float, U: np.ndarray) -> float:
        """
        单样本：给定重排后全状态 x_full（W 输出顺序 [前车v,自车v,后2v,后3v,后4v, 前车s,自车s,后2s,后3s,后4s]）、
        期望、控制序列 U，返回总代价。
        """
        dt = self.dt
        Q_s, Q_v, R = self.Q_s, self.Q_v, self.R
        Q_vel_lead = self.Q_vel_lead
        Q_s_lead = self.Q_s_lead
        Q_fs, Q_fv = self.Q_following_s, self.Q_following_v
        Q_st, Q_vt = self.Q_s_term, self.Q_v_term
        N = len(U)
        # 重排后 W 输出: [v0_v, v1_v, v2_v, v3_v, v4_v, v0_s, v1_s, v2_s, v3_s, v4_s]
        # 即 [前车v, 自车v, 后2v, 后3v, 后4v, 前车s, 自车s, 后2s, 后3s, 后4s]
        s_star = 20
        v_star = 15
        s_lead = float(x_full[5])   # 前车 spacing
        v_lead = float(x_full[0])   # 前车速度 (原 x_full[9] 是后车4的s，错误)
        s_f2, v_f2 = float(x_full[7]), float(x_full[2])  # 后车2: s在7, v在2
        s_f3, v_f3 = float(x_full[8]), float(x_full[3])
        s_f4, v_f4 = float(x_full[9]), float(x_full[4])

        cost = 0.0
        cost += Q_fs * ((s_f2 - s_star) ** 2 + (s_f3 - s_star) ** 2 + (s_f4 - s_star) ** 2)
        cost += Q_fv * ((v_f2 - v_star) ** 2 + (v_f3 - v_star) ** 2 + (v_f4 - v_star) ** 2)

        s, v = float(x_full[6]), float(x_full[1])  # 自车: spacing 在 6, velocity 在 1
        for k in range(N):
            cost += Q_s * (s - s_star) ** 2 + Q_v * (v - v_star) ** 2
            cost += Q_vel_lead * (v - v_lead) ** 2 + Q_s_lead * (s - s_lead) ** 2
            cost += R * (U[k] - u_star) ** 2
            s = s + dt * (v_lead - v)
            v = v + dt * U[k]
        cost += Q_st * (s - s_star) ** 2 + Q_vt * (v - v_star) ** 2
        cost += Q_vel_lead * (v - v_lead) ** 2 + Q_s_lead * (s - s_lead) ** 2
        return cost

    def _solve_mpc(self, x_full: np.ndarray, x_star_full: np.ndarray, u_star: float, u_min: float, u_max: float) -> float:
        """对单样本求解 MPC，返回 u_0。"""
        N = self.horizon
        u_min, u_max = float(u_min), float(u_max)

        def obj(U_flat):
            return self._rollout_and_cost(x_full, x_star_full, u_star, U_flat)

        bounds = [(u_min, u_max)] * N
        U0 = np.zeros(N)
        res = minimize(obj, U0, method="L-BFGS-B", bounds=bounds)
        u0 = float(res.x[0]) if res.success else 0.0
        return np.clip(u0, u_min, u_max)

    def forward(self, x, x_star, u_star, u_bounds):
        u_min, u_max = u_bounds
        u_min = u_min if isinstance(u_min, (int, float)) else u_min.to(self.device)
        u_max = u_max if isinstance(u_max, (int, float)) else u_max.to(self.device)
        x = x.reshape(-1, self.state_dim).to(self.device)
        x_star = x_star.reshape(-1, self.state_dim).to(self.device)
        x = x @ self.W_change_state_position
        x_star = x_star @ self.W_change_state_position
        x_full_np = x.cpu().numpy()
        x_star_full_np = x_star.cpu().numpy()
        u_star = u_star.reshape(-1, 1).to(self.device)
        u_star_np = u_star.cpu().numpy()
        u_min_np = np.array(u_min, dtype=np.float64) if isinstance(u_min, (int, float)) else u_min.detach().cpu().numpy()
        u_max_np = np.array(u_max, dtype=np.float64) if isinstance(u_max, (int, float)) else u_max.detach().cpu().numpy()
        batch_size = x_full_np.shape[0]
        u0_list = []
        for b in range(batch_size):
            u_min_b = float(u_min_np) if u_min_np.size == 1 else float(u_min_np.flat[b])
            u_max_b = float(u_max_np) if u_max_np.size == 1 else float(u_max_np.flat[b])
            u0_b = self._solve_mpc(
                x_full_np[b], x_star_full_np[b], float(u_star_np[b, 0]), u_min_b, u_max_b,
            )
            u0_list.append(u0_b)
        u = torch.tensor(np.array(u0_list, dtype=np.float32).reshape(-1, 1), device=self.device, dtype=torch.float32)
        return u


def get_mpc_controller(
    state_dim: int = 10,
    control_dim: int = 1,
    dt: float = 0.1,
    horizon: int = 10,
    Q_spacing: float = 1.0,
    Q_velocity: float = 1.0,
    R_control: float = 0.1,
    device: str = None,
) -> MPCController:
    """构造 MPC 控制器实例。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return MPCController(
        state_dim=state_dim,
        control_dim=control_dim,
        dt=dt,
        horizon=horizon,
        Q_spacing=Q_spacing,
        Q_velocity=Q_velocity,
        R_control=R_control,
        device=device,
    )

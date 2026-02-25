import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
from training_exp_comb import PlatoonDynamics
import torch.nn as nn
from networks import NetworkController, system_network, DoubleQCritic, VectorLyapunovNetwork, system_network
from model_based_control import LinearFeedbackController, MPCController
import os
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化系统参数：第一段 5 车 [Leading, CAV, HDV, HDV, HDV]，后续每段 4 车 [CAV, HDV, HDV, HDV]
num_vehicles = 5   # 控制器输入为 5 车状态 (state_dim=10)
# 5 + 4*6 = 29 车：第一段 5 辆，后面 6 段各 4 辆
following_cav = 6
num_vehicles_sim = 5 + 4 * following_cav  # 29
# CAV 位置：第一段第 2 辆 index=1；后续每段第 1 辆 index=5,9,13,17,21,25
cav_indices = [1] + [5 + 4 * k for k in range(following_cav)]  # [1, 5, 9, 13, 17, 21, 25]
dynamics_params = {
    'dt': 0.1,
    'alpha': 6,
    'beta': 9,
    'v_max': 30.0,
    's_st': 5.0,
    's_go': 35.0,
    'a_max': 100.0,
    'a_min': -100.0,
    'desired_spacing': 20.0
}

# 设置全局样式
#plt.style.use('seaborn-white')  # 使用清爽的背景样式

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24  # 设置默认字体大小
plt.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 24  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 24  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 24  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 24  # 图例字体大小



# 创建连接矩阵
connection_matrix = {i: {} for i in range(num_vehicles_sim)}
for i in range(1, num_vehicles_sim):
    if i in cav_indices:
        connection_matrix[i][i-1] = 0.5
    else:
        connection_matrix[i][i-1] = 1.0

# 初始化系统动力学
system_dynamics_network = system_network(state_dim=3).to(device)
system = PlatoonDynamics(dynamics_params, connection_matrix, False, system_dynamics_network)

# 加载参数并分离控制器参数
check_point = torch.load(f'model_weights/best_model-v925.ckpt') #best_model-v912/925   best_model_iss-v1.ckpt
parameters = check_point['state_dict']
# 重新映射参数键名
pre_trained_id = 95
if_load_pre_trained_model = False
if if_load_pre_trained_model:
    pre_trained_model = f"pre_train_model/sac_platoon_{pre_trained_id}_actor.pth"
    raw_parameters = torch.load(pre_trained_model)
    controller_parameters = {}
    for k, v in raw_parameters.items():
        if k.startswith('trunk'):
            new_key = k.replace('trunk', 'network')
            # 如果是最后一层的参数，只取一半（对应均值输出）

            if 'network.4.weight' in new_key:  
                controller_parameters[new_key] = v[:1, :]  # 只保留第一行，对应均值
            elif 'network.4.bias' in new_key:
                controller_parameters[new_key] = v[:1]  # 只保留第一个元素，对应均值
            else:
                controller_parameters[new_key] = v

    print("Original keys:", raw_parameters.keys())
    print("New keys:", controller_parameters.keys())
else:
    controller_parameters = {}
    for k, v in parameters.items():
        if k.startswith('controllers'):
            new_key = k.replace('controllers.', '')
            controller_parameters[new_key] = v

# 控制器列表长度 = 仿真车数，仅 CAV(index=1) 为网络控制器，输入为前 5 车状态 (10 维)
controllers = nn.ModuleList([
    NetworkController(10, 1).to(device) if i in cav_indices
    else nn.Identity() for i in range(num_vehicles_sim)
])

pre_trained_model = "pre_train_model/sac_platoon_"+str(pre_trained_id)+"_actor.pth"
raw_parameters = torch.load(pre_trained_model)
original_controller_parameters = {}
for k, v in raw_parameters.items():
    if k.startswith('trunk'):
        new_key = k.replace('trunk', '1.network')
        # 如果是最后一层的参数，只取一半（对应均值输出）

        if '1.network.4.weight' in new_key:  
            original_controller_parameters[new_key] = v[:1, :]  # 只保留第一行，对应均值
        elif '1.network.4.bias' in new_key:
            original_controller_parameters[new_key] = v[:1]  # 只保留第一个元素，对应均值
        else:
            original_controller_parameters[new_key] = v

original_controllers = nn.ModuleList([
    NetworkController(10, 1).to(device) if i in cav_indices
    else nn.Identity() for i in range(num_vehicles_sim)
])
original_controller_parameters = {k.replace('1.', ''): v for k, v in original_controller_parameters.items()}
for idx in cav_indices:
    original_controllers[idx].load_state_dict(original_controller_parameters)

# 模型基控制器：LCC 与 MPC（输入均为前 5 车状态）
model_based_controllers = nn.ModuleList([
    LinearFeedbackController(10, 1, device=device) if i in cav_indices
    else nn.Identity() for i in range(num_vehicles_sim)
])
model_based_controllers.eval()

# horizon 过短会导致 MPC 只顾短期、spacing 易持续偏离；提高 Q_spacing 强化对间距的惩罚
mpc_controllers = nn.ModuleList([
    MPCController(10, 1, dt=dynamics_params['dt'], horizon=3, device=device) if i in cav_indices
    else nn.Identity() for i in range(num_vehicles_sim)
])
mpc_controllers.eval()

# 将 checkpoint 中 CAV 参数加载到所有 CAV 控制器（每块共用一个网络）
if not if_load_pre_trained_model:
    controller_parameters_cav = {k.replace('1.', ''): v for k, v in controller_parameters.items() if k.startswith('1.')}
    for idx in cav_indices:
        controllers[idx].load_state_dict(controller_parameters_cav)
else:
    for i in range(num_vehicles_sim):
        if i in cav_indices:
            controllers[i].load_state_dict(controller_parameters)
controllers.eval()

critics = DoubleQCritic(10, 1).to(device)
pre_trained_critics = f"pre_train_model/sac_platoon_{pre_trained_id}_critic.pth"
if pre_trained_critics is not None:
    raw_parameters_critics = torch.load(pre_trained_critics)
    critics.load_state_dict(raw_parameters_critics)

# 初始化状态（30 车）
batch_size = 1
states = torch.zeros((batch_size, num_vehicles_sim, 2)).to(device)
states[:, 0, 0] = 20
states[:, 0, 1] = 15.0
for i in range(1, num_vehicles_sim):
    states[:, i, 0] = 20.0
    states[:, i, 1] = 15.0
states_original = states.clone()

# 存储轨迹
time_steps = 500
trajectories = [states.clone().to(device)]
trajectories_original = [states_original.clone().to(device)]
states_model_based = states.clone()
trajectories_model_based = [states_model_based.clone().to(device)]
states_mpc = states.clone()
trajectories_mpc = [states_mpc.clone().to(device)]
disturbances = torch.zeros((batch_size, num_vehicles_sim)).to(device)

# 模拟系统                                                                                                                                                                  
with torch.no_grad():
    for t in range(time_steps):
        # 领头车扰动：从最高点 10 按 sin 曲线下降至 0（前 15 步）
        if t <= 15:
            disturbances[:, 0] = 7 * torch.sin(torch.tensor(2 * np.pi * t / 15))

        # 计算控制输入：每个 CAV 使用「前车+本车+后 3 车」共 5 车状态 (state_dim=10)
        controls = []
        controls_original = []
        controls_model_based = []
        controls_mpc = []
        for i in range(num_vehicles_sim):
            if i in cav_indices:
                # 5 车块：前车 i-1 与 本车及后车 i..i+3
                block_start = i - 1
                state_local = states[:, block_start:block_start + num_vehicles, :]
                state_orig_local = states_original[:, block_start:block_start + num_vehicles, :]
                state_mb_local = states_model_based[:, block_start:block_start + num_vehicles, :]
                state_mpc_local = states_mpc[:, block_start:block_start + num_vehicles, :]
                x_star = torch.tensor([20.0, 15.0] * 5).to(device)
                u_star = torch.zeros(1).to(device)
                u_bounds = (torch.tensor(-5.0), torch.tensor(5.0).to(device))

                control = controllers[i](state_local, x_star, u_star, u_bounds)
                control_original = original_controllers[i](state_orig_local, x_star, u_star, u_bounds)
                control_model_based = model_based_controllers[i](state_mb_local, x_star, u_star, u_bounds)
                control_mpc = mpc_controllers[i](state_mpc_local, x_star, u_star, u_bounds)
                controls.append(control)
                controls_original.append(control_original)
                controls_model_based.append(control_model_based)
                controls_mpc.append(control_mpc)
            else:
                controls.append(None)
                controls_original.append(None)
                controls_model_based.append(None)
                controls_mpc.append(None)
        
        # 更新状态
        states = system.next_state(states, controls, disturbances)
        states_original = system.next_state(states_original, controls_original, disturbances)
        states_model_based = system.next_state(states_model_based, controls_model_based, disturbances)
        states_mpc = system.next_state(states_mpc, controls_mpc, disturbances)
        trajectories.append(states.clone())
        trajectories_original.append(states_original.clone())
        trajectories_model_based.append(states_model_based.clone())
        trajectories_mpc.append(states_mpc.clone())

# 转换为numpy数组进行绘图
trajectories = torch.stack(trajectories).squeeze(1).cpu().numpy()
trajectories_original = torch.stack(trajectories_original).squeeze(1).cpu().numpy()
trajectories_model_based = torch.stack(trajectories_model_based).squeeze(1).cpu().numpy()
trajectories_mpc = torch.stack(trajectories_mpc).squeeze(1).cpu().numpy()

# 绘制轨迹（第一段 5 车，后续每段 4 车）
# 图例用：Leading / CAV / HDV 三类（其余图 fig2 等仍用每车一色）
labels = []
for i in range(num_vehicles_sim):
    if i == 0:
        labels.append('Leading Vehicle')
    elif i in cav_indices:
        labels.append(f'CAV{1 + (i - 1) // 4}')
    else:
        labels.append(f'HDV{i}')
cmap = plt.cm.get_cmap('viridis', num_vehicles_sim)
colors = [cmap(i) for i in range(num_vehicles_sim)]

# fig1: 按类型上色，图例只 3 条
fig1 = plt.figure(figsize=(8, 6), dpi=300)
ax1 = fig1.add_subplot(2, 1, 1)
ax2 = fig1.add_subplot(2, 1, 2)
color_leading = 'k'
color_cav = 'C0'
color_hdv = 'C1'
colors_fig1 = [color_leading if i == 0 else (color_cav if i in cav_indices else color_hdv) for i in range(num_vehicles_sim)]

for i in range(num_vehicles_sim):
    spacing = trajectories[:, i, 0]
    velocities = trajectories[:, i, 1]
    ax1.plot(spacing, color=colors_fig1[i])
    ax2.plot(velocities, color=colors_fig1[i])
ax1.set_ylabel('Spacing (m)', fontsize=24)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax2.set_ylabel('Velocity (m/s)', fontsize=24)
ax2.set_xlabel('Time Steps', fontsize=24)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
# 简洁图例：仅 3 类
leg_handles = [
    Line2D([0], [0], color=color_leading, lw=2, label='Leading Vehicle'),
    Line2D([0], [0], color=color_cav, lw=2, label='CAV'),
    Line2D([0], [0], color=color_hdv, lw=2, label='HDV'),
]
ax1.legend(handles=leg_handles, loc='best', fontsize=14)
fig1.savefig('output_figures/trajectory.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig1)

# fig2: Original 控制器，同 fig1 的三类图例
fig2 = plt.figure(figsize=(8, 6), dpi=300)
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)
for i in range(num_vehicles_sim):
    spacing = trajectories_original[:, i, 0]
    velocities = trajectories_original[:, i, 1]
    ax1.plot(spacing, color=colors_fig1[i])
    ax2.plot(velocities, color=colors_fig1[i])
ax1.set_ylabel('Spacing (m)', fontsize=24)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax2.set_ylabel('Velocity (m/s)', fontsize=24)
ax2.set_xlabel('Time Steps', fontsize=24)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax1.legend(handles=leg_handles, loc='best', fontsize=14)
fig2.savefig('output_figures/trajectory_original.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig2)

# 线性反馈模型控制器轨迹（LCC），三类图例
fig_mb = plt.figure(figsize=(8, 6), dpi=300)
ax1 = fig_mb.add_subplot(2, 1, 1)
ax2 = fig_mb.add_subplot(2, 1, 2)
for i in range(num_vehicles_sim):
    spacing = trajectories_model_based[:, i, 0]
    velocities = trajectories_model_based[:, i, 1]
    ax1.plot(spacing, color=colors_fig1[i])
    ax2.plot(velocities, color=colors_fig1[i])
ax1.set_ylabel('Spacing (m)', fontsize=24)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax2.set_ylabel('Velocity (m/s)', fontsize=24)
ax2.set_xlabel('Time Steps', fontsize=24)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax1.legend(handles=leg_handles, loc='best', fontsize=14)
fig_mb.savefig('output_figures/trajectory_model_based.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig_mb)

# MPC 控制器轨迹，三类图例
fig_mpc = plt.figure(figsize=(8, 6), dpi=300)
ax1 = fig_mpc.add_subplot(2, 1, 1)
ax2 = fig_mpc.add_subplot(2, 1, 2)
for i in range(num_vehicles_sim):
    spacing = trajectories_mpc[:, i, 0]
    velocities = trajectories_mpc[:, i, 1]
    ax1.plot(spacing, color=colors_fig1[i])
    ax2.plot(velocities, color=colors_fig1[i])
ax1.set_ylabel('Spacing (m)', fontsize=24)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax2.set_ylabel('Velocity (m/s)', fontsize=24)
ax2.set_xlabel('Time Steps', fontsize=24)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax1.legend(handles=leg_handles, loc='best', fontsize=14)
fig_mpc.savefig('output_figures/trajectory_mpc.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig_mpc)

# visualize lyapunov functions（仍用 5 车网络与 5x5 连接矩阵）
spacing_space = np.linspace(15, 25, 100)
velocity_space = np.linspace(10, 20, 100)

G = torch.zeros(num_vehicles, num_vehicles).to(device)
for i in range(1, num_vehicles):
    G[i, i - 1] = 1.0

for vehicle_idx in range(2):#num_vehicles-1
    V = np.zeros((len(spacing_space), len(velocity_space)))

    state_dims = [2] * num_vehicles
    V_net = VectorLyapunovNetwork(state_dim=state_dims, G=G, device=device).to(device)
    V_parameters = {}
    for k, v in parameters.items():
        if k.startswith('V_net.'):
            new_key = k.replace('V_net.', '')
            V_parameters[new_key] = v
    print(V_parameters.keys())
    print(V_net)
    V_net.load_state_dict(V_parameters)

    for i, s in enumerate(spacing_space):
        for j, v in enumerate(velocity_space):
            if vehicle_idx == 0:
                x = torch.tensor([[20.0, 15.0, s, v, 20.0, 15.0, 20.0, 15.0, 20.0, 15.0]], dtype=torch.float32).to(device)
                x_star = torch.tensor([[20.0, 15.0]*5], dtype=torch.float32).to(device)
                V[i, j] = V_net(x, x_star)[0][0].item()
            else:
                x = torch.tensor([[20.0, 15.0, 20.0, 15.0, s, v, 20.0, 15.0, 20.0, 15.0]], dtype=torch.float32).to(device)
                x_star = torch.tensor([[20.0, 15.0]*5], dtype=torch.float32).to(device)
                V[i, j] = V_net(x, x_star)[0][1].item()

    # Create a meshgrid: X corresponds to spacing, Y corresponds to velocity
    X, Y = np.meshgrid(spacing_space, velocity_space)

    # Since you used V.T in contourf, Z would be V.T to match X, Y shapes
    Z = V.T

    # 3D Lyapunov图
    try:
        fig2 = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig2.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=True)
        ax.set_xlabel('Spacing (m)', fontsize=24, labelpad=10)
        ax.set_ylabel('Velocity (m/s)', fontsize=24, labelpad=10)
        ax.set_zlabel('Lyapunov Function', fontsize=24, labelpad=10)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.tick_params(axis='z', labelsize=24)
        cbar_3d = fig2.colorbar(surf, ax=ax, shrink=0.6, format=mtick.FormatStrFormatter('%.2e'))
        cbar_3d.ax.tick_params(labelsize=24)
        cbar_3d.set_label('Lyapunov Function', fontsize=24)
        ax.view_init(elev=30, azim=45)  # 优化视角
        plt.tight_layout()
        if vehicle_idx == 0:
            fig2.savefig('output_figures/lyapunov_3d_CAV.pdf', format='pdf', bbox_inches='tight', dpi=300)
        else:
            fig2.savefig('output_figures/lyapunov_3d_HDV.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig2)
    except Exception as e:
        pass

    # 2D等高线图
    fig3, ax = plt.subplots(figsize=(8, 6), dpi=300)
    contour = ax.contourf(X, Y, Z, cmap='viridis', levels=20, alpha=0.95)
    ax.plot(20, 15, 'r*', markersize=14, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Spacing (m)', fontsize=26, labelpad=10)
    ax.set_ylabel('Velocity (m/s)', fontsize=26, labelpad=10)
    ax.tick_params(axis='both', labelsize=24)
    cbar = plt.colorbar(contour, ax=ax, format=mtick.FormatStrFormatter('%.1e'))
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Lyapunov Function', fontsize=24)
    plt.legend(fontsize=24, frameon=True, fancybox=True, framealpha=0.8)
    plt.tight_layout()
    
    if vehicle_idx == 0:
        fig3.savefig('output_figures/lyapunov_contour_CAV.pdf', format='pdf', bbox_inches='tight', dpi=300)
    else:
        fig3.savefig('output_figures/lyapunov_contour_HDV.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig3)

values_new_controller = np.zeros((len(spacing_space), len(velocity_space)))
value_origin_controller = np.zeros((len(spacing_space), len(velocity_space)))

for i, s in enumerate(spacing_space):
    for j, v in enumerate(velocity_space):
        sample_x = torch.tensor([[20.0, 15.0,s, v, 20.0, 15.0, 20.0, 15.0, 20.0, 15.0]],dtype=torch.float32).to(device)

        x_star = torch.tensor([[20.0, 15.0]*5],dtype=torch.float32).to(device)
        u_star = torch.zeros(1).to(device)
        u_bounds = (torch.tensor(-5.0), torch.tensor(5.0).to(device))
        new_controller = controllers[1](sample_x, x_star, u_star, u_bounds)
        origin_controller = original_controllers[1](sample_x, x_star, u_star, u_bounds)
        values_new_controller[i,j] = critics(sample_x, new_controller).item()
        value_origin_controller[i,j] = critics(sample_x, origin_controller).item()

# Create a meshgrid
X, Y = np.meshgrid(spacing_space, velocity_space)

# Q-value差值图
try:
    fig4 = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig4.add_subplot(111, projection='3d')
    surf_1 = ax.plot_surface(X, Y, values_new_controller-value_origin_controller, 
                            cmap='viridis', antialiased=True)
    ax.set_xlabel('Spacing (m)', fontsize=16, labelpad=10)
    ax.set_ylabel('Velocity (m/s)', fontsize=16, labelpad=10)
    ax.set_zlabel('Q-value difference', fontsize=16, labelpad=10)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    ax.view_init(elev=30, azim=45)  # 优化视角
    plt.tight_layout()
    fig4.savefig('output_figures/q_value_difference_3d.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig4)
except Exception as e:
    pass

# 新增：Q-value差值的等高线图
fig5, ax = plt.subplots(figsize=(8, 6), dpi=300)
contour = ax.contourf(X, Y, values_new_controller-value_origin_controller, 
                     cmap='viridis', levels=20, alpha=0.95)
ax.plot(20, 15, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
ax.set_xlabel('Spacing (m)', fontsize=16, labelpad=10)
ax.set_ylabel('Velocity (m/s)', fontsize=16, labelpad=10)
ax.tick_params(axis='both', labelsize=14)
cbar_q = plt.colorbar(contour, ax=ax, format=mtick.FormatStrFormatter('%.2e'))
cbar_q.ax.tick_params(labelsize=14)
cbar_q.set_label('Q-value difference', fontsize=16)
plt.legend(fontsize=14, frameon=True, fancybox=True, framealpha=0.8)
plt.tight_layout()
fig5.savefig('output_figures/q_value_difference_contour.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig5)

print(np.sum(np.abs(trajectories_original[:,1,0]-trajectories[:,1,0])))

# calculate tracking error for the CAV (all controllers)，速度相对前车
leading_vel = trajectories[:, 0, 1]  # 前车（领头车）速度，CAV1 与之比较
cav_vel_new = trajectories[:, 1, 1]  # CAV velocity with new controller
cav_vel_original = trajectories_original[:, 1, 1]  # CAV velocity with original controller
cav_vel_model_based = trajectories_model_based[:, 1, 1]  # CAV velocity with LCC
cav_vel_mpc = trajectories_mpc[:, 1, 1]  # CAV velocity with MPC

# Velocity errors relative to preceding vehicle (leading vehicle for CAV1)
velocity_error_new = cav_vel_new - leading_vel
velocity_error_original = cav_vel_original - leading_vel
velocity_error_model_based = cav_vel_model_based - leading_vel
velocity_error_mpc = cav_vel_mpc - leading_vel

# Calculate error statistics
mean_error_new = np.mean(np.abs(velocity_error_new))
mean_error_original = np.mean(np.abs(velocity_error_original))
mean_error_model_based = np.mean(np.abs(velocity_error_model_based))
mean_error_mpc = np.mean(np.abs(velocity_error_mpc))
max_error_new = np.max(np.abs(velocity_error_new))
max_error_original = np.max(np.abs(velocity_error_original))
max_error_model_based = np.max(np.abs(velocity_error_model_based))
max_error_mpc = np.max(np.abs(velocity_error_mpc))
rmse_new = np.sqrt(np.mean(velocity_error_new**2))
rmse_original = np.sqrt(np.mean(velocity_error_original**2))
rmse_model_based = np.sqrt(np.mean(velocity_error_model_based**2))
rmse_mpc = np.sqrt(np.mean(velocity_error_mpc**2))

print("\nCAV Velocity Tracking Error Relative to Preceding Vehicle (Leading):")
print("-" * 125)
print(f"{'Metric':<30} {'New':<12} {'Original':<12} {'LCC (model-based)':<20} {'MPC (model-based)':<20}")
print("-" * 125)
print(f"{'Mean Absolute Error (m/s)':<30} {mean_error_new:<12.4f} {mean_error_original:<12.4f} {mean_error_model_based:<20.4f} {mean_error_mpc:<20.4f}")
print(f"{'Maximum Absolute Error (m/s)':<30} {max_error_new:<12.4f} {max_error_original:<12.4f} {max_error_model_based:<20.4f} {max_error_mpc:<20.4f}")
print(f"{'RMSE (m/s)':<30} {rmse_new:<12.4f} {rmse_original:<12.4f} {rmse_model_based:<20.4f} {rmse_mpc:<20.4f}")
print(f"{'Sum of Squared Errors':<30} {np.sum(velocity_error_new**2):<12.4f} {np.sum(velocity_error_original**2):<12.4f} {np.sum(velocity_error_model_based**2):<20.4f} {np.sum(velocity_error_mpc**2):<20.4f}")
print("-" * 125)

# Calculate tracking error sum for all vehicles (velocity: relative to leading vehicle)
all_vehicles_error_new = 0
all_vehicles_error_original = 0
all_vehicles_error_model_based = 0
all_vehicles_error_mpc = 0

# 头车速度/间距作为参考（所有车都和头车比）
leading_vel_new = trajectories[:, 0, 1]
leading_vel_orig = trajectories_original[:, 0, 1]
leading_vel_mb = trajectories_model_based[:, 0, 1]
leading_vel_mpc = trajectories_mpc[:, 0, 1]
leading_spacing_new = trajectories[:, 0, 0]
leading_spacing_orig = trajectories_original[:, 0, 0]
leading_spacing_mb = trajectories_model_based[:, 0, 0]
leading_spacing_mpc = trajectories_mpc[:, 0, 0]

print("\nVelocity Tracking Error Summary for All Vehicles (relative to leading vehicle, $v_i - v_0$):")
print("-" * 115)
print(f"{'Vehicle':<10} {'New RMSE':<12} {'Original RMSE':<14} {'LCC RMSE':<12} {'MPC RMSE':<12}")
print("-" * 115)

for i in range(1, num_vehicles_sim):  # Skip the leading vehicle (i=0)
    vel_error_new = trajectories[:, i, 1] - leading_vel_new
    vel_error_original = trajectories_original[:, i, 1] - leading_vel_orig
    vel_error_model_based = trajectories_model_based[:, i, 1] - leading_vel_mb
    vel_error_mpc = trajectories_mpc[:, i, 1] - leading_vel_mpc
    rmse_new = np.sqrt(np.mean(vel_error_new**2))
    rmse_original = np.sqrt(np.mean(vel_error_original**2))
    rmse_model_based = np.sqrt(np.mean(vel_error_model_based**2))
    rmse_mpc = np.sqrt(np.mean(vel_error_mpc**2))
    vehicle_type = f"CAV{1 + (i - 1) // 4}" if i in cav_indices else f"HDV{i}"
    print(f"{vehicle_type:<10} {rmse_new:<12.4f} {rmse_original:<14.4f} {rmse_model_based:<12.4f} {rmse_mpc:<12.4f}")
    all_vehicles_error_new += rmse_new
    all_vehicles_error_original += rmse_original
    all_vehicles_error_model_based += rmse_model_based
    all_vehicles_error_mpc += rmse_mpc

print("-" * 115)
print(f"{'Total':<10} {all_vehicles_error_new:<12.4f} {all_vehicles_error_original:<14.4f} {all_vehicles_error_model_based:<12.4f} {all_vehicles_error_mpc:<12.4f}")
print("-" * 115)

# Spacing error：所有车都和头车比，$s_i - s_0$
print("\nSpacing Error Summary for All Vehicles (relative to leading vehicle, $s_i - s_0$, RMSE in m):")
print("-" * 115)
print(f"{'Vehicle':<10} {'New RMSE':<12} {'Original RMSE':<14} {'LCC RMSE':<12} {'MPC RMSE':<12}")
print("-" * 115)
all_spacing_error_new = 0
all_spacing_error_original = 0
all_spacing_error_model_based = 0
all_spacing_error_mpc = 0
for i in range(1, num_vehicles_sim):
    sp_error_new = trajectories[:, i, 0] - leading_spacing_new
    sp_error_original = trajectories_original[:, i, 0] - leading_spacing_orig
    sp_error_model_based = trajectories_model_based[:, i, 0] - leading_spacing_mb
    sp_error_mpc = trajectories_mpc[:, i, 0] - leading_spacing_mpc
    rmse_new = np.sqrt(np.mean(sp_error_new**2))
    rmse_original = np.sqrt(np.mean(sp_error_original**2))
    rmse_model_based = np.sqrt(np.mean(sp_error_model_based**2))
    rmse_mpc = np.sqrt(np.mean(sp_error_mpc**2))
    vehicle_type = f"CAV{1 + (i - 1) // 4}" if i in cav_indices else f"HDV{i}"
    print(f"{vehicle_type:<10} {rmse_new:<12.4f} {rmse_original:<14.4f} {rmse_model_based:<12.4f} {rmse_mpc:<12.4f}")
    all_spacing_error_new += rmse_new
    all_spacing_error_original += rmse_original
    all_spacing_error_model_based += rmse_model_based
    all_spacing_error_mpc += rmse_mpc
print("-" * 115)
print(f"{'Total':<10} {all_spacing_error_new:<12.4f} {all_spacing_error_original:<14.4f} {all_spacing_error_model_based:<12.4f} {all_spacing_error_mpc:<12.4f}")
print("-" * 115)

# --- Velocity Gain (L∞: 时间上最大绝对值之比) ---
# G_{v,i} = ||v_i||_∞ / ||v_{i-1}||_∞,  ||v||_∞ = max_t |v(t)|
print("\nVelocity Gain $G_{v,i}$ in the $\\mathcal{L}_\\infty$ sense (max over time):")
print("-" * 95)
print(f"{'Vehicle':<10} {'New':<12} {'Original':<12} {'LCC':<12} {'MPC':<12}")
print("-" * 95)
for i in range(1, num_vehicles_sim):
    den_new = np.max(np.abs(trajectories[:, i-1, 1]))
    gain_new = np.max(np.abs(trajectories[:, i, 1])) / den_new if den_new > 0 else np.nan
    den_orig = np.max(np.abs(trajectories_original[:, i-1, 1]))
    gain_orig = np.max(np.abs(trajectories_original[:, i, 1])) / den_orig if den_orig > 0 else np.nan
    den_mb = np.max(np.abs(trajectories_model_based[:, i-1, 1]))
    gain_model_based = np.max(np.abs(trajectories_model_based[:, i, 1])) / den_mb if den_mb > 0 else np.nan
    den_mpc = np.max(np.abs(trajectories_mpc[:, i-1, 1]))
    gain_mpc = np.max(np.abs(trajectories_mpc[:, i, 1])) / den_mpc if den_mpc > 0 else np.nan
    vehicle_type = f"CAV{1 + (i - 1) // 4}" if i in cav_indices else f"HDV{i}"
    print(f"{vehicle_type:<10} {gain_new:<12.4f} {gain_orig:<12.4f} {gain_model_based:<12.4f} {gain_mpc:<12.4f}")
print("-" * 95)

# --- Spacing Gain (L∞: 相对平衡点的偏差的最大值之比) ---
# G_{s,i} = ||s_i - s*||_∞ / ||s_{i-1} - s*||_∞,  s* = desired_spacing
s_eq = dynamics_params.get('desired_spacing', 20.0)
print("\nSpacing Gain $G_{s,i}$ in the $\\mathcal{L}_\\infty$ sense (max over time, relative to equilibrium $s^*$):")
print("-" * 95)
print(f"{'Vehicle':<10} {'New':<12} {'Original':<12} {'LCC':<12} {'MPC':<12}")
print("-" * 95)
for i in range(1, num_vehicles_sim):
    den_new = np.max(np.abs(trajectories[:, i-1, 0] - s_eq))
    gain_new = np.max(np.abs(trajectories[:, i, 0] - s_eq)) / den_new if den_new > 0 else np.nan
    den_orig = np.max(np.abs(trajectories_original[:, i-1, 0] - s_eq))
    gain_orig = np.max(np.abs(trajectories_original[:, i, 0] - s_eq)) / den_orig if den_orig > 0 else np.nan
    den_mb = np.max(np.abs(trajectories_model_based[:, i-1, 0] - s_eq))
    gain_model_based = np.max(np.abs(trajectories_model_based[:, i, 0] - s_eq)) / den_mb if den_mb > 0 else np.nan
    den_mpc = np.max(np.abs(trajectories_mpc[:, i-1, 0] - s_eq))
    gain_mpc = np.max(np.abs(trajectories_mpc[:, i, 0] - s_eq)) / den_mpc if den_mpc > 0 else np.nan
    vehicle_type = f"CAV{1 + (i - 1) // 4}" if i in cav_indices else f"HDV{i}"
    print(f"{vehicle_type:<10} {gain_new:<12.4f} {gain_orig:<12.4f} {gain_model_based:<12.4f} {gain_mpc:<12.4f}")
print("-" * 95)

# --- Combined Gain (L∞: 归一化后时间上最大联合量之比) ---
# 归一化 s_norm, v_norm; 联合量 m(t)=sqrt(s_norm^2+v_norm^2), ||m||_∞=max_t m(t); gain=当前车/前车
s_ref = dynamics_params.get('desired_spacing', 20.0)
v_ref = 15.0
def _combined_norm(traj, veh_idx, s_ref, v_ref):
    s = (traj[:, veh_idx, 0] - s_ref) / s_ref
    v = (traj[:, veh_idx, 1] - v_ref) / v_ref
    return np.max(np.sqrt(s**2 + v**2))

print("\nCombined Gain $G_{sv,i}$ in the $\\mathcal{L}_\\infty$ sense (normalized spacing & velocity, max over time):")
print("-" * 95)
print(f"{'Vehicle':<10} {'New':<12} {'Original':<12} {'LCC':<12} {'MPC':<12}")
print("-" * 95)
for i in range(1, num_vehicles_sim):
    den_new = _combined_norm(trajectories, i - 1, s_ref, v_ref)
    gain_new = _combined_norm(trajectories, i, s_ref, v_ref) / den_new if den_new > 0 else np.nan
    den_orig = _combined_norm(trajectories_original, i - 1, s_ref, v_ref)
    gain_orig = _combined_norm(trajectories_original, i, s_ref, v_ref) / den_orig if den_orig > 0 else np.nan
    den_mb = _combined_norm(trajectories_model_based, i - 1, s_ref, v_ref)
    gain_model_based = _combined_norm(trajectories_model_based, i, s_ref, v_ref) / den_mb if den_mb > 0 else np.nan
    den_mpc = _combined_norm(trajectories_mpc, i - 1, s_ref, v_ref)
    gain_mpc = _combined_norm(trajectories_mpc, i, s_ref, v_ref) / den_mpc if den_mpc > 0 else np.nan
    vehicle_type = f"CAV{1 + (i - 1) // 4}" if i in cav_indices else f"HDV{i}"
    print(f"{vehicle_type:<10} {gain_new:<12.4f} {gain_orig:<12.4f} {gain_model_based:<12.4f} {gain_mpc:<12.4f}")
print("-" * 95)

# --- 各控制器「所有车中最大 gain」---
# 对每种 gain（velocity / spacing / combined）取 max over vehicles (i=1..N-1)
gv_new, gv_orig, gv_mb, gv_mpc = [], [], [], []
gs_new, gs_orig, gs_mb, gs_mpc = [], [], [], []
gsv_new, gsv_orig, gsv_mb, gsv_mpc = [], [], [], []
for i in range(1, num_vehicles_sim):
    den = np.max(np.abs(trajectories[:, i-1, 1]))
    gv_new.append(np.max(np.abs(trajectories[:, i, 1])) / den if den > 0 else np.nan)
    den = np.max(np.abs(trajectories_original[:, i-1, 1]))
    gv_orig.append(np.max(np.abs(trajectories_original[:, i, 1])) / den if den > 0 else np.nan)
    den = np.max(np.abs(trajectories_model_based[:, i-1, 1]))
    gv_mb.append(np.max(np.abs(trajectories_model_based[:, i, 1])) / den if den > 0 else np.nan)
    den = np.max(np.abs(trajectories_mpc[:, i-1, 1]))
    gv_mpc.append(np.max(np.abs(trajectories_mpc[:, i, 1])) / den if den > 0 else np.nan)

    den = np.max(np.abs(trajectories[:, i-1, 0] - s_eq))
    gs_new.append(np.max(np.abs(trajectories[:, i, 0] - s_eq)) / den if den > 0 else np.nan)
    den = np.max(np.abs(trajectories_original[:, i-1, 0] - s_eq))
    gs_orig.append(np.max(np.abs(trajectories_original[:, i, 0] - s_eq)) / den if den > 0 else np.nan)
    den = np.max(np.abs(trajectories_model_based[:, i-1, 0] - s_eq))
    gs_mb.append(np.max(np.abs(trajectories_model_based[:, i, 0] - s_eq)) / den if den > 0 else np.nan)
    den = np.max(np.abs(trajectories_mpc[:, i-1, 0] - s_eq))
    gs_mpc.append(np.max(np.abs(trajectories_mpc[:, i, 0] - s_eq)) / den if den > 0 else np.nan)

    den = _combined_norm(trajectories, i - 1, s_ref, v_ref)
    gsv_new.append(_combined_norm(trajectories, i, s_ref, v_ref) / den if den > 0 else np.nan)
    den = _combined_norm(trajectories_original, i - 1, s_ref, v_ref)
    gsv_orig.append(_combined_norm(trajectories_original, i, s_ref, v_ref) / den if den > 0 else np.nan)
    den = _combined_norm(trajectories_model_based, i - 1, s_ref, v_ref)
    gsv_mb.append(_combined_norm(trajectories_model_based, i, s_ref, v_ref) / den if den > 0 else np.nan)
    den = _combined_norm(trajectories_mpc, i - 1, s_ref, v_ref)
    gsv_mpc.append(_combined_norm(trajectories_mpc, i, s_ref, v_ref) / den if den > 0 else np.nan)

print("\nMax Gain over All Vehicles (per controller):")
print("-" * 95)
print(f"{'Gain':<12} {'New':<18} {'Original':<18} {'LCC':<18} {'MPC':<18}")
print("-" * 95)
print(f"{'max $G_{v,i}$':<12} {np.nanmax(gv_new):<18.4f} {np.nanmax(gv_orig):<18.4f} {np.nanmax(gv_mb):<18.4f} {np.nanmax(gv_mpc):<18.4f}")
print(f"{'max $G_{s,i}$':<12} {np.nanmax(gs_new):<18.4f} {np.nanmax(gs_orig):<18.4f} {np.nanmax(gs_mb):<18.4f} {np.nanmax(gs_mpc):<18.4f}")
print(f"{'max $G_{sv,i}$':<12} {np.nanmax(gsv_new):<18.4f} {np.nanmax(gsv_orig):<18.4f} {np.nanmax(gsv_mb):<18.4f} {np.nanmax(gsv_mpc):<18.4f}")
print("-" * 95)

# 头尾 string stability：整条链 gain 乘起来 Π G_i
print("\nHead-to-Tail (product of gains $\\prod_{i=1}^{N-1} G_i$):")
print("-" * 95)
print(f"{'Product':<14} {'New':<18} {'Original':<18} {'LCC':<18} {'MPC':<18}")
print("-" * 95)
print(f"{'$\\prod G_{v,i}$':<14} {np.nanprod(gv_new):<18.4e} {np.nanprod(gv_orig):<18.4e} {np.nanprod(gv_mb):<18.4e} {np.nanprod(gv_mpc):<18.4e}")
print(f"{'$\\prod G_{s,i}$':<14} {np.nanprod(gs_new):<18.4e} {np.nanprod(gs_orig):<18.4e} {np.nanprod(gs_mb):<18.4e} {np.nanprod(gs_mpc):<18.4e}")
print(f"{'$\\prod G_{sv,i}$':<14} {np.nanprod(gsv_new):<18.4e} {np.nanprod(gsv_orig):<18.4e} {np.nanprod(gsv_mb):<18.4e} {np.nanprod(gsv_mpc):<18.4e}")
print("-" * 95)


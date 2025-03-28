import matplotlib.pyplot as plt
import numpy as np
import torch
from training_exp_comb import PlatoonDynamics
import torch.nn as nn
from networks import NetworkController, system_network, DoubleQCritic, VectorLyapunovNetwork, system_network
import os

# 初始化系统参数
num_vehicles = 5
cav_indices = [1]  # 第二辆车是CAV
dynamics_params = {
    'dt': 0.1,
    'alpha': 0.6,
    'beta': 0.9,
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
plt.rcParams['font.size'] = 12  # 设置默认字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小



# 创建连接矩阵
connection_matrix = {i: {} for i in range(num_vehicles)}
for i in range(1, num_vehicles):
    if i in cav_indices:
        connection_matrix[i][i-1] = 0.5
    else:
        connection_matrix[i][i-1] = 1.0

# 初始化系统动力学
system_dynamics_network = system_network(state_dim=3)
system = PlatoonDynamics(dynamics_params, connection_matrix, True, system_dynamics_network)

# 加载参数并分离控制器参数
check_point = torch.load(f'model_weights/best_model-v908.ckpt') #229
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

controllers = nn.ModuleList([
    NetworkController(10, 1) if i in cav_indices  # state_dim=2, control_dim=1
    else nn.Identity() for i in range(num_vehicles)
])

#print("controller_parameters", controller_parameters)
#print("controllers", controllers)
if not if_load_pre_trained_model:   
    controllers.load_state_dict(controller_parameters)
else:
    for i in range(num_vehicles):
        if i in cav_indices:
            controllers[i].load_state_dict(controller_parameters)
controllers.eval()

critics = DoubleQCritic(10, 1)
pre_trained_critics = f"pre_train_model/sac_platoon_{pre_trained_id}_critic.pth"
if pre_trained_critics is not None:
    raw_parameters_critics = torch.load(pre_trained_critics)
    critics.load_state_dict(raw_parameters_critics)

# 初始化状态
batch_size = 1
states = torch.zeros((batch_size, num_vehicles, 2))
# 设置初始状态
states[:, 0, 0] = 20  # 领头车位置
states[:, 0, 1] = 15.0  # 领头车速度
for i in range(1, num_vehicles):
    states[:, i, 0] = 20.0  # 每辆车间隔20米
    states[:, i, 1] = 15.0  # 初始速度

# 存储轨迹
time_steps = 1000
trajectories = [states.clone()]
disturbances = torch.zeros((batch_size, num_vehicles))

# 模拟系统
with torch.no_grad():
    for t in range(time_steps):
        # 为领头车添加正弦扰动
        if t<=100:
            disturbances[:, 0] = 3.0 * torch.sin(torch.tensor(2 * np.pi * t / 50))  # 振幅2.0，周期50步
        
        # 计算控制输入
        controls = []
        for i in range(num_vehicles):
            if i in cav_indices:
                state_i = states[:, i, :]
                x_star = torch.tensor([20.0, 15.0]*5)  # 期望状态
                u_star = torch.zeros(1)
                u_bounds = (torch.tensor(-5.0), torch.tensor(5.0))

                control = controllers[i](states, x_star, u_star, u_bounds)
                controls.append(control)#
            else:
                controls.append(None)
        
        # 更新状态
        states = system.next_state(states, controls, disturbances)
        trajectories.append(states.clone())

# 转换为numpy数组进行绘图
trajectories = torch.stack(trajectories).squeeze(1).numpy()

# 绘制轨迹
fig1 = plt.figure(figsize=(8, 6), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 蓝色、橙色、绿色、红色、紫色
labels = ['Leading Vehicle', 'CAV1', 'HDV1', 'HDV2', 'HDV3']

for i in range(num_vehicles):
    spacing = trajectories[:, i, 0]
    velocities = trajectories[:, i, 1]
    
    plt.subplot(2, 1, 1)
    plt.plot(spacing, label=labels[i], color=colors[i])
    plt.ylabel('Spacing (m)', fontsize=14)
    # Increase grid clarity
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.subplot(2, 1, 2)
    plt.plot(velocities, label=labels[i], color=colors[i])
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.xlabel('Time Steps', fontsize=14)
    # Increase grid clarity
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12)

plt.tight_layout()
fig1.savefig('output_figures/trajectory.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig1)

# visualize lyapunov functions
spacing_space = np.linspace(15, 25, 100)
velocity_space = np.linspace(10, 20, 100)

G = torch.zeros(len(system.connections), len(system.connections))
for i in system.connections:
    for j in system.connections[i]:
        G[i, j] = 1.0

for vehicle_idx in range(2):#num_vehicles-1
    V = np.zeros((len(spacing_space), len(velocity_space)))

    state_dims = [2] * num_vehicles
    V_net = VectorLyapunovNetwork(state_dim=state_dims, G=G)
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
                x = torch.tensor([[20.0, 15.0, s, v, 20.0, 15.0, 20.0, 15.0, 20.0, 15.0]], dtype=torch.float32)
                x_star = torch.tensor([[20.0, 15.0]*5], dtype=torch.float32)
                V[i, j] = V_net(x, x_star)[0][0].item()
            else:
                x = torch.tensor([[20.0, 15.0, 20.0, 15.0, s, v, 20.0, 15.0, 20.0, 15.0]], dtype=torch.float32)
                x_star = torch.tensor([[20.0, 15.0]*5], dtype=torch.float32)
                V[i, j] = V_net(x, x_star)[0][1].item()

    # Create a meshgrid: X corresponds to spacing, Y corresponds to velocity
    X, Y = np.meshgrid(spacing_space, velocity_space)

    # Since you used V.T in contourf, Z would be V.T to match X, Y shapes
    Z = V.T

    # 3D Lyapunov图
    fig2 = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig2.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=True)
    ax.set_xlabel('Spacing (m)', fontsize=14, labelpad=10)
    ax.set_ylabel('Velocity (m/s)', fontsize=14, labelpad=10)
    ax.set_zlabel('Lyapunov Function', fontsize=14, labelpad=10)
    ax.view_init(elev=30, azim=45)  # 优化视角
    plt.tight_layout()
    if vehicle_idx == 0:
        fig2.savefig('output_figures/lyapunov_3d_CAV.pdf', format='pdf', bbox_inches='tight', dpi=300)
    else:
        fig2.savefig('output_figures/lyapunov_3d_HDV.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig2)

    # 2D等高线图
    fig3, ax = plt.subplots(figsize=(8, 6), dpi=300)
    contour = ax.contourf(X, Y, Z, cmap='viridis', levels=20, alpha=0.95)
    ax.plot(20, 15, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Spacing (m)', fontsize=14, labelpad=10)
    ax.set_ylabel('Velocity (m/s)', fontsize=14, labelpad=10)
    plt.colorbar(contour)  # 添加颜色条
    plt.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.8)
    plt.tight_layout()
    
    if vehicle_idx == 0:
        fig3.savefig('output_figures/lyapunov_contour_CAV.pdf', format='pdf', bbox_inches='tight', dpi=300)
    else:
        fig3.savefig('output_figures/lyapunov_contour_HDV.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig3)

values_new_controller = np.zeros((len(spacing_space), len(velocity_space)))
value_origin_controller = np.zeros((len(spacing_space), len(velocity_space)))

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
    NetworkController(10, 1) if i in cav_indices  # state_dim=2, control_dim=1
    else nn.Identity() for i in range(num_vehicles)
])
original_controller_parameters = {k.replace('1.', ''): v for k, v in original_controller_parameters.items()}
original_controllers[1].load_state_dict(original_controller_parameters)

for i, s in enumerate(spacing_space):
    for j, v in enumerate(velocity_space):
        sample_x = torch.tensor([[20.0, 15.0,s, v, 20.0, 15.0, 20.0, 15.0, 20.0, 15.0]],dtype=torch.float32)

        x_star = torch.tensor([[20.0, 15.0]*5],dtype=torch.float32)
        u_star = torch.zeros(1)
        u_bounds = (torch.tensor(-5.0), torch.tensor(5.0))
        new_controller = controllers[1](sample_x, x_star, u_star, u_bounds)
        origin_controller = original_controllers[1](sample_x, x_star, u_star, u_bounds)
        values_new_controller[i,j] = critics(sample_x, new_controller).item()
        value_origin_controller[i,j] = critics(sample_x, origin_controller).item()

# Create a meshgrid
X, Y = np.meshgrid(spacing_space, velocity_space)

# Q-value差值图
fig4 = plt.figure(figsize=(8, 6), dpi=300)
ax = fig4.add_subplot(111, projection='3d')
surf_1 = ax.plot_surface(X, Y, values_new_controller-value_origin_controller, 
                        cmap='viridis', antialiased=True)
ax.set_xlabel('Spacing (m)', fontsize=14, labelpad=10)
ax.set_ylabel('Velocity (m/s)', fontsize=14, labelpad=10)
ax.set_zlabel('Q-value difference', fontsize=14, labelpad=10)
ax.view_init(elev=30, azim=45)  # 优化视角
plt.tight_layout()
fig4.savefig('output_figures/q_value_difference_3d.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig4)

# 新增：Q-value差值的等高线图
fig5, ax = plt.subplots(figsize=(8, 6), dpi=300)
contour = ax.contourf(X, Y, values_new_controller-value_origin_controller, 
                     cmap='viridis', levels=20, alpha=0.95)
ax.plot(20, 15, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
ax.set_xlabel('Spacing (m)', fontsize=14, labelpad=10)
ax.set_ylabel('Velocity (m/s)', fontsize=14, labelpad=10)
plt.colorbar(contour)  # 添加颜色条
plt.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.8)
plt.tight_layout()
fig5.savefig('output_figures/q_value_difference_contour.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close(fig5)
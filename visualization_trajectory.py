import matplotlib.pyplot as plt
import numpy as np
import torch
from training_exp_comb import PlatoonDynamics, NetworkController
import torch.nn as nn

# 初始化系统参数
num_vehicles = 3
cav_indices = [1]  # 第二辆车是CAV
dynamics_params = {
    'dt': 0.1,
    'alpha': 0.6,
    'beta': 0.9,
    'v_max': 30.0,
    's_st': 5.0,
    's_go': 35.0,
    'a_max': 5.0,
    'a_min': -5.0,
    'desired_spacing': 20.0
}

# 创建连接矩阵
connection_matrix = {i: {} for i in range(num_vehicles)}
for i in range(1, num_vehicles):
    if i in cav_indices:
        connection_matrix[i][i-1] = 0.5
    else:
        connection_matrix[i][i-1] = 1.0

# 初始化系统动力学
system = PlatoonDynamics(dynamics_params, connection_matrix)

# 加载参数并分离控制器参数
check_point = torch.load(f'model_weights/last.ckpt')
parameters = check_point['state_dict']
# 重新映射参数键名
controller_parameters = {}
for k, v in parameters.items():
    if k.startswith('controllers.1.network'):
        # 从 'controllers.1.network.0.weight' 转换为 '1.network.0.weight'
        new_key = k.replace('controllers.', '')
        controller_parameters[new_key] = v

controllers = nn.ModuleList([
    NetworkController(2, 1) if i in cav_indices  # state_dim=2, control_dim=1
    else nn.Identity() for i in range(num_vehicles)
])


controllers.load_state_dict(controller_parameters)
controllers.eval()

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
        disturbances[:, 0] = 2.0 * torch.sin(torch.tensor(2 * np.pi * t / 50))  # 振幅2.0，周期50步
        
        # 计算控制输入
        controls = []
        for i in range(num_vehicles):
            if i in cav_indices:
                state_i = states[:, i, :]
                x_star = torch.tensor([20.0, 15.0])  # 期望状态
                u_star = torch.zeros(1)
                u_bounds = (torch.tensor(-5.0), torch.tensor(5.0))
                control = controllers[i](state_i, x_star, u_star, u_bounds)
                controls.append(control)
            else:
                controls.append(None)
        
        # 更新状态
        states = system.next_state(states, controls, disturbances)
        trajectories.append(states.clone())

# 转换为numpy数组进行绘图
trajectories = torch.stack(trajectories).squeeze(1).numpy()

# 绘制轨迹
plt.figure(figsize=(12, 8))
colors = ['b', 'r', 'g']
labels = ['Leading Vehicle', 'CAV', 'HDV']

for i in range(num_vehicles):
    spacing = trajectories[:, i, 0]
    velocities = trajectories[:, i, 1]
    
    plt.subplot(2, 1, 1)
    plt.plot(spacing, label=labels[i], color=colors[i])
    plt.ylabel('spacing (m)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(velocities, label=labels[i], color=colors[i])
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time Steps')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()




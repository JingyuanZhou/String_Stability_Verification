import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation  # ✅ 正确
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
from networks import VectorLyapunovNetwork
from pre_train_model.learn_dynamics_control import DynamicsNN, ControllerNN
from networks import CombinedController
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize system parameters  
num_uavs = 3  # Total number of UAVs: 1 leader + 2 followers
controlled_indices = [1, 2]  # Both follower UAVs are controlled
dynamics_params = {
    'dt': 0.1,
    'max_thrust': 5.0,    # Maximum thrust in any direction
    'min_thrust': -5.0,   # Minimum thrust in any direction
    'max_velocity': 30.0, # Maximum velocity magnitude in any direction
    'desired_spacing': np.array([10.0, 0.0, 0.0])  # Desired spacing in 3D
}

# Set global plotting style
#plt.style.use('seaborn-white')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24  # 设置默认字体大小
plt.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 24  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 24  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 24  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 24  # 图例字体大小

# Create connection matrix (influence of preceding UAV on follower)
connection_matrix = {i: {} for i in range(num_uavs)}
for i in range(1, num_uavs):
    if i in controlled_indices:
        connection_matrix[i][i-1] = 0.5  # Each UAV is influenced by the one in front
    else:
        connection_matrix[i][i-1] = 1.0

# Initialize system dynamics with appropriate dimensionality
dim = 3  # 3D space
state_dim = 2 * dim  # Position and velocity
system_dynamics_network = DynamicsNN(state_dim=state_dim, action_dim=dim).to(device)

# Load model from checkpoint
check_point = torch.load('model_weights/best_uav_model-v45.ckpt')  #best_uav_model-v2 best_uav_model-v45 best_uav_model_mode_ISS-v1
parameters = check_point['state_dict']

# Extract controller parameters for the CombinedController
controller_parameters = {}
original_controller_parameters = {}
controller1_params = {}
controller2_params = {}
original_controller1_params = {}
original_controller2_params = {}

for k, v in parameters.items():
    if k.startswith('controller.controller_1'):
        new_key = k.replace('controller.controller_1.', '')
        controller1_params[new_key] = v
    if k.startswith('controller.controller_2'):
        new_key = k.replace('controller.controller_2.', '')
        controller2_params[new_key] = v
    if k.startswith('original_controller.controller_1'):
        new_key = k.replace('original_controller.controller_1.', '')
        original_controller1_params[new_key] = v
    if k.startswith('original_controller.controller_2'):
        new_key = k.replace('original_controller.controller_2.', '')
        original_controller2_params[new_key] = v

# Create CombinedController
combined_controller = CombinedController(input_dim=state_dim, output_dim=dim).to(device)
combined_controller.controller_1.load_state_dict(controller1_params)
combined_controller.controller_2.load_state_dict(controller2_params)
combined_controller.eval()

original_combined_controller = CombinedController(input_dim=state_dim, output_dim=dim).to(device)
original_combined_controller.controller_1.load_state_dict(original_controller1_params)
original_combined_controller.controller_2.load_state_dict(original_controller2_params)
original_combined_controller.eval()

# Initialize state
batch_size = 1
states = torch.zeros((batch_size, num_uavs, state_dim)).to(device)

# Set initial states for all UAVs
# Leader position [0,0,0] and velocity [5,0,0]
states[:, 0, 0:3] = torch.tensor([0.0, 0.0, 0.0]).to(device)  # Leader position
states[:, 0, 3:6] = torch.tensor([5.0, 0.0, 0.0]).to(device)  # Leader velocity

# First follower position behind leader, same velocity
states[:, 1, 0:3] = torch.tensor([-10.0, 0.0, 0.0]).to(device)  # First follower position
states[:, 1, 3:6] = torch.tensor([5.0, 0.0, 0.0]).to(device)    # First follower velocity

# Second follower position behind first follower, same velocity
states[:, 2, 0:3] = torch.tensor([-20.0, 0.0, 0.0]).to(device)  # Second follower position
states[:, 2, 3:6] = torch.tensor([5.0, 0.0, 0.0]).to(device)    # Second follower velocity

# Initialize original state
original_states = states.clone()

# Store trajectories
time_steps = 300
trajectories = [states.clone()]
disturbances = torch.zeros((batch_size, num_uavs, dim)).to(device)
original_trajectories = [original_states.clone()]

# Simulate system with 3D disturbances - similar to simulate_UAVs approach
cruise_velocity = torch.tensor([5.0, 0.0, 0.0]).to(device)
dist_amplitude = torch.tensor([3, 0.0, 0.0]).to(device)  # Similar to simulate_UAVs.py torch.tensor([0.2, 0.5, 0.3]) 
dist_frequency = torch.tensor([0.4, 0.0, 0.0]).to(device)  # Match frequencies from simulate_UAVs.py torch.tensor([0.5, 0.3, 0.4])

with torch.no_grad():
    for t in range(time_steps):
        time_value = t * dynamics_params['dt']  # Current time value
        
        # Extract current state
        current_state = trajectories[-1]
        original_current_state = original_trajectories[-1]
        next_state = torch.zeros_like(current_state).to(device)
        original_next_state = torch.zeros_like(original_current_state).to(device)

        # Leader UAV dynamics - similar to simulate_UAVs approach
        # 1. Calculate target velocity (cruise velocity + sinusoidal disturbance)
        target_velocity = cruise_velocity.clone()
        for j in range(dim):
            target_velocity[j] += dist_amplitude[j] * torch.sin(dist_frequency[j] * time_value)
        
        # 2. Calculate target position
        target_position = torch.zeros(dim).to(device)
        for j in range(dim):
            # Base position from cruise velocity
            target_position[j] = cruise_velocity[j] * time_value
            # Add position offset from sinusoidal disturbance
            if dist_frequency[j] > 0:
                target_position[j] += (dist_amplitude[j] / dist_frequency[j]) * (1 - torch.cos(dist_frequency[j] * time_value))
        
        # 3. Calculate position and velocity errors
        position_error = target_position - current_state[0, 0, 0:3]
        velocity_error = target_velocity - current_state[0, 0, 3:6]
        original_position_error = target_position - original_current_state[0, 0, 0:3]
        original_velocity_error = target_velocity - original_current_state[0, 0, 3:6]
        
        # 4. PD control for leader
        leader_control = 2.0 * position_error + 1.0 * velocity_error
        original_leader_control = 2.0 * original_position_error + 1.0 * original_velocity_error

        # Update leader position based on current velocity
        next_state[:, 0, 0:3] = current_state[:, 0, 0:3] + current_state[:, 0, 3:6] * dynamics_params['dt']
        original_next_state[:, 0, 0:3] = original_current_state[:, 0, 0:3] + original_current_state[:, 0, 3:6] * dynamics_params['dt']

        # Update leader velocity using PD control
        next_state[:, 0, 3:6] = current_state[:, 0, 3:6] + leader_control * dynamics_params['dt']
        original_next_state[:, 0, 3:6] = original_current_state[:, 0, 3:6] + original_leader_control * dynamics_params['dt']

        # Loop through each follower UAV
        for follower_idx in controlled_indices:
            # Rest of the follower control code remains the same
            preceding_idx = follower_idx - 1
            p_ref = current_state[:, preceding_idx, 0:3] - torch.tensor(dynamics_params['desired_spacing']).to(device)
            v_ref = current_state[:, preceding_idx, 3:6]
            original_p_ref = original_current_state[:, preceding_idx, 0:3] - torch.tensor(dynamics_params['desired_spacing']).to(device)
            original_v_ref = original_current_state[:, preceding_idx, 3:6]
            
            p_current = current_state[:, follower_idx, 0:3]
            v_current = current_state[:, follower_idx, 3:6]
            original_p_current = original_current_state[:, follower_idx, 0:3]
            original_v_current = original_current_state[:, follower_idx, 3:6]
            
            p_error = p_ref - p_current
            v_error = v_ref - v_current
            original_p_error = original_p_ref - original_p_current
            original_v_error = original_v_ref - original_v_current
            if follower_idx == 1:
                control = combined_controller.controller_1(torch.cat((p_error, v_error), dim=-1).to(dtype=torch.float32))
                original_control = original_combined_controller.controller_1(torch.cat((original_p_error, original_v_error), dim=-1).to(dtype=torch.float32))
            else:
                control = combined_controller.controller_2(torch.cat((p_error, v_error), dim=-1).to(dtype=torch.float32))
                original_control = original_combined_controller.controller_2(torch.cat((original_p_error, original_v_error), dim=-1).to(dtype=torch.float32))
            next_state[:, follower_idx, 0:3] = current_state[:, follower_idx, 0:3] + current_state[:, follower_idx, 3:6] * dynamics_params['dt']
            next_state[:, follower_idx, 3:6] = current_state[:, follower_idx, 3:6] + control * dynamics_params['dt']
            original_next_state[:, follower_idx, 0:3] = original_current_state[:, follower_idx, 0:3] + original_current_state[:, follower_idx, 3:6] * dynamics_params['dt']
            original_next_state[:, follower_idx, 3:6] = original_current_state[:, follower_idx, 3:6] + original_control * dynamics_params['dt']
        
        trajectories.append(next_state)
        original_trajectories.append(original_next_state)

# Prepare data for visualization
trajectory_array = torch.stack(trajectories).squeeze(1).cpu().numpy()
original_trajectory_array = torch.stack(original_trajectories).squeeze(1).cpu().numpy()
time_array = np.arange(time_steps + 1) * dynamics_params['dt']

# Create output directory if needed
os.makedirs('output_figures', exist_ok=True)
colors = ['b', 'r', 'g']
labels = ['Leader UAV', 'First Follower', 'Second Follower']
try:
    # Plot 3D trajectory
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Plot all UAVs' trajectories with different colors

    for i in range(num_uavs):
        uav_traj = trajectory_array[:, i, 0:3]
        ax.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], f'{colors[i]}-', linewidth=2, label=labels[i])

    # Add markers at specific time points
    marker_indices = np.linspace(0, time_steps, 10, dtype=int)
    for idx in marker_indices:
        # Get positions at this time point for all UAVs
        for i in range(num_uavs):
            x, y, z = trajectory_array[idx, i, 0:3]
            ax.scatter(x, y, z, c=colors[i], s=50, marker='o')
        
        # Add time label (only on leader for clarity)
        time_val = idx * dynamics_params['dt']
        lx, ly, lz = trajectory_array[idx, 0, 0:3]
        ax.text(lx, ly, lz+0.5, f't={time_val:.1f}s', fontsize=8)
        
        # Connection lines between UAVs at this time
        for i in range(1, num_uavs):
            prev_x, prev_y, prev_z = trajectory_array[idx, i-1, 0:3]
            curr_x, curr_y, curr_z = trajectory_array[idx, i, 0:3]
            ax.plot([prev_x, curr_x], [prev_y, curr_y], [prev_z, curr_z], 'k--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D UAV Formation Flight Trajectory')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=45)  # Optimize viewpoint

    plt.savefig('output_figures/uav_3d_trajectory.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
except Exception as e:
    pass
# Plot position components over time
fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)

# Get trajectories for each UAV
uav_trajectories = [trajectory_array[:, i, 0:3] for i in range(num_uavs)]

# Plot each position component
for ax_idx, component in enumerate(['X', 'Y', 'Z']):
    for uav_idx in range(num_uavs):
        axs[ax_idx].plot(time_array, uav_trajectories[uav_idx][:, ax_idx], 
                         f'{colors[uav_idx]}-', linewidth=2, label=labels[uav_idx])
    
    axs[ax_idx].set_ylabel(f'{component} Position (m)')
    axs[ax_idx].grid(True)
    
    # Only add legend to the first subplot
    if ax_idx == 0:
        axs[ax_idx].legend()

axs[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('output_figures/uav_positions_time.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot formation errors over time
fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)

# Define colors for different followers
follower_colors = ['r', 'g']
follower_labels = ['First Follower Error', 'Second Follower Error']

# Calculate errors for each dimension and each follower
desired_spacing = dynamics_params['desired_spacing']

for dim_idx, component in enumerate(['X', 'Y', 'Z']):
    for follower_idx in range(1, num_uavs):
        # Calculate error between this follower and its preceding UAV
        preceding_idx = follower_idx - 1
        error = (uav_trajectories[preceding_idx][:, dim_idx] - 
                 uav_trajectories[follower_idx][:, dim_idx] - 
                 desired_spacing[dim_idx])
        
        axs[dim_idx].plot(time_array, error, 
                          f'{follower_colors[follower_idx-1]}-', 
                          linewidth=2, 
                          label=follower_labels[follower_idx-1])
    
    axs[dim_idx].set_ylabel(f'{component} Error (m)')
    axs[dim_idx].grid(True)
    axs[dim_idx].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Only add legend to the first subplot
    if dim_idx == 0:
        axs[dim_idx].legend()

axs[2].set_xlabel('Time (s)')

plt.suptitle('Formation Error Over Time')
plt.tight_layout()
plt.savefig('output_figures/uav_errors_time.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# Extract Lyapunov network parameters
V_params = {}
V_network1_params = {}
V_network2_params = {}

for k, v in parameters.items():
    if k.startswith('V_net.network_1'):
        new_key = k.replace('V_net.network_1.', '')
        V_network1_params[new_key] = v
    if k.startswith('V_net.network_2'):
        new_key = k.replace('V_net.network_2.', '')
        V_network2_params[new_key] = v

# Initialize Lyapunov network with 3 UAVs
V_net = VectorLyapunovNetwork(input_dim=2*dim, hidden_dim=64, G=None).to(device)
V_net.num_UAVs = 3

# Create a coupling matrix G for visualization
G = torch.zeros(num_uavs, num_uavs)
for i in range(1, num_uavs):
    G[i, i-1] = connection_matrix[i][i-1]

V_net.coupling_matrix = G

# Load parameters for each network
# We need to handle loading parameters correctly based on the actual network structure

# Try loading network parameters - might need to adjust this based on actual structure
state_dict = {}
for k, v in V_network1_params.items():
    state_dict[f'network_1.{k}'] = v
for k, v in V_network2_params.items():
    state_dict[f'network_2.{k}'] = v

# Check if we have all required keys
existing_keys = set(state_dict.keys())
required_keys = set(dict(V_net.named_parameters()).keys())
missing_keys = required_keys - existing_keys

if missing_keys:
    print(f"Warning: Missing keys in state dict: {missing_keys}")
    # Initialize missing keys randomly
    for k in missing_keys:
        if 'weight' in k:
            # Initialize weights with small random values
            shape = getattr(V_net, k.split('.')[0])[int(k.split('.')[1])].weight.shape
            state_dict[k] = torch.randn(shape) * 0.01
        elif 'bias' in k:
            # Initialize biases to zero
            shape = getattr(V_net, k.split('.')[0])[int(k.split('.')[1])].bias.shape
            state_dict[k] = torch.zeros(shape)

V_net.load_state_dict(state_dict)


V_net.eval()

delta_ref = dynamics_params['desired_spacing'][0]  # X-direction spacing

# Create a grid for x-position error and x-velocity error
x_error_space = np.linspace(-3.0, 3.0, 100)  # X-position error range
vx_error_space = np.linspace(-3.0, 3.0, 100)  # X-velocity error range

# Initialize Lyapunov values for both followers
V1 = np.zeros((len(x_error_space), len(vx_error_space)))
V2 = np.zeros((len(x_error_space), len(vx_error_space)))

# Compute Lyapunov values over the grid
for i, p_err in enumerate(x_error_space):
    for j, v_err in enumerate(vx_error_space):
        # Create a full 6D error state vector with zeros for y and z dimensions
        # Format: [px_err, py_err, pz_err, vx_err, vy_err, vz_err]
        x = torch.tensor([p_err, 0.0, 0.0, v_err, 0.0, 0.0, p_err, 0.0, 0.0, v_err, 0.0, 0.0], dtype=torch.float32).to(device)
        
        # Compute Lyapunov values for each follower
        with torch.no_grad():
            # For first follower (UAV 1)
            V1[i, j], V2[i, j] = V_net(x)
            

# Create meshgrid for 3D plotting
X, Y = np.meshgrid(x_error_space, vx_error_space)
Z1 = V1.T  # Transpose to match meshgrid dimensions
Z2 = V2.T  # Transpose to match meshgrid dimensions

# 3D Lyapunov plot for first follower
try:
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z1, cmap='viridis', antialiased=True)
    ax.set_xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
    ax.set_ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
    ax.set_zlabel('Lyapunov Function (UAV 1)', fontsize=14, labelpad=10)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig('output_figures/uav1_lyapunov_3d.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # 3D Lyapunov plot for second follower
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z2, cmap='plasma', antialiased=True)
    ax.set_xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
    ax.set_ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
    ax.set_zlabel('Lyapunov Function (UAV 2)', fontsize=14, labelpad=10)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig('output_figures/uav2_lyapunov_3d.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
except Exception as e:
    pass

# 2D contour plots
# First follower individual plot
fig = plt.figure(figsize=(8, 6), dpi=300)
contour1 = plt.contourf(X, Y, Z1, cmap='viridis', levels=20, alpha=0.95)
plt.plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
plt.xlabel('Position Error (m)', fontsize=24, labelpad=10)
plt.ylabel('Velocity Error (m/s)', fontsize=24, labelpad=10)
#plt.title('Lyapunov Function (UAV 1)', fontsize=24)
plt.colorbar(contour1, label='Lyapunov Value')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('output_figures/uav1_lyapunov_2d.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# Second follower individual plot
fig = plt.figure(figsize=(8, 6), dpi=300)
contour2 = plt.contourf(X, Y, Z2, cmap='viridis', levels=20, alpha=0.95)
plt.plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
plt.xlabel('Position Error (m)', fontsize=24, labelpad=10)
plt.ylabel('Velocity Error (m/s)', fontsize=24, labelpad=10)
#plt.title('Lyapunov Function (UAV 2)', fontsize=24)
plt.colorbar(contour2, label='Lyapunov Value')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('output_figures/uav2_lyapunov_2d.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# Combined 2D contour plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

# Follower 1 contour
contour1 = axes[0].contourf(X, Y, Z1, cmap='viridis', levels=20, alpha=0.95)
axes[0].plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
axes[0].set_xlabel('Position Error (m)', fontsize=24, labelpad=10)
axes[0].set_ylabel('Velocity Error (m/s)', fontsize=24, labelpad=10)
#axes[0].set_title('Lyapunov Function (UAV 1)', fontsize=24)
plt.colorbar(contour1, ax=axes[0], label='Lyapunov Value')
axes[0].legend()
axes[0].grid(True)

# Follower 2 contour
contour2 = axes[1].contourf(X, Y, Z2, cmap='plasma', levels=20, alpha=0.95)
axes[1].plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
axes[1].set_xlabel('Position Error (m)', fontsize=24, labelpad=10)
axes[1].set_ylabel('Velocity Error (m/s)', fontsize=24, labelpad=10)
#axes[1].set_title('Lyapunov Function (UAV 2)', fontsize=24)
plt.colorbar(contour2, ax=axes[1], label='Lyapunov Value')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('output_figures/uav_lyapunov_contours.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# Calculate velocity tracking error between the following UAVs and the leading UAV
# Extract velocity components for all UAVs
leader_velocity = trajectory_array[:, 0, 3:6]  # Leader velocity [vx, vy, vz]
follower1_velocity = trajectory_array[:, 1, 3:6]  # First follower velocity
follower2_velocity = trajectory_array[:, 2, 3:6]  # Second follower velocity

# For velocity tracking, followers should match leader's velocity
# Calculate velocity errors (actual velocity - ideal velocity)
follower1_velocity_error = np.linalg.norm(follower1_velocity - leader_velocity, axis=1)
follower2_velocity_error = np.linalg.norm(follower2_velocity - leader_velocity, axis=1)

# Calculate original velocity tracking errors for comparison
original_leader_velocity = original_trajectory_array[:, 0, 3:6]
original_follower1_velocity = original_trajectory_array[:, 1, 3:6]
original_follower2_velocity = original_trajectory_array[:, 2, 3:6]

original_follower1_velocity_error = np.linalg.norm(original_follower1_velocity - original_leader_velocity, axis=1)
original_follower2_velocity_error = np.linalg.norm(original_follower2_velocity - original_leader_velocity, axis=1)

# Calculate RMSE (Root Mean Square Error)
follower1_rmse = np.sqrt(np.mean(np.square(follower1_velocity_error)))
follower2_rmse = np.sqrt(np.mean(np.square(follower2_velocity_error)))
original_follower1_rmse = np.sqrt(np.mean(np.square(original_follower1_velocity_error)))
original_follower2_rmse = np.sqrt(np.mean(np.square(original_follower2_velocity_error)))

# Calculate sum of RMSE for the UAV platoon
fine_tuned_sum_rmse = follower1_rmse + follower2_rmse
original_sum_rmse = original_follower1_rmse + original_follower2_rmse

# Calculate percentage of enhancement
enhancement_percentage = ((original_sum_rmse - fine_tuned_sum_rmse) / original_sum_rmse) * 100

# Calculate statistics for the velocity tracking errors
velocity_error_stats = {
    'Follower 1': {
        'Mean Velocity Error': np.mean(follower1_velocity_error),
        'Max Velocity Error': np.max(follower1_velocity_error),
        'Std Dev Velocity': np.std(follower1_velocity_error),
        'RMSE Velocity': follower1_rmse
    },
    'Follower 2': {
        'Mean Velocity Error': np.mean(follower2_velocity_error),
        'Max Velocity Error': np.max(follower2_velocity_error),
        'Std Dev Velocity': np.std(follower2_velocity_error),
        'RMSE Velocity': follower2_rmse
    },
    'Original Follower 1': {
        'Mean Velocity Error': np.mean(original_follower1_velocity_error),
        'Max Velocity Error': np.max(original_follower1_velocity_error),
        'Std Dev Velocity': np.std(original_follower1_velocity_error),
        'RMSE Velocity': original_follower1_rmse
    },
    'Original Follower 2': {
        'Mean Velocity Error': np.mean(original_follower2_velocity_error),
        'Max Velocity Error': np.max(original_follower2_velocity_error),
        'Std Dev Velocity': np.std(original_follower2_velocity_error),
        'RMSE Velocity': original_follower2_rmse
    }
}

print("\nUAV Formation Velocity Tracking Error Statistics:")
for uav, stats in velocity_error_stats.items():
    print(f"\n{uav}:")
    for metric, value in stats.items():
        print(f"  {metric}: {value:.4f} m/s")

print("\nOverall UAV Platoon Performance:")
print(f"  Fine-tuned Sum RMSE: {fine_tuned_sum_rmse:.4f} m/s")
print(f"  Original Sum RMSE: {original_sum_rmse:.4f} m/s")
print(f"  Performance Enhancement: {enhancement_percentage:.2f}%")


import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
from networks import VectorLyapunovNetwork
from pre_train_model.learn_dynamics_control import DynamicsNN, ControllerNN
from networks import CombinedController
import os

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
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

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
system_dynamics_network = DynamicsNN(state_dim=state_dim, action_dim=dim)

# Load model from checkpoint
check_point = torch.load('model_weights/best_uav_model-v108.ckpt')
parameters = check_point['state_dict']
print(parameters.keys())


# Extract Lyapunov network parameters
V_params = {}
V_network1_params = {}
V_network2_params = {}
V_network3_params = {}

for k, v in parameters.items():
    if k.startswith('V_net.networks.0'):
        new_key = k.replace('V_net.networks.0.', '')
        V_network1_params[new_key] = v
    if k.startswith('V_net.networks.1'):
        new_key = k.replace('V_net.networks.1.', '')
        V_network2_params[new_key] = v
    if k.startswith('V_net.networks.2'):
        new_key = k.replace('V_net.networks.2.', '')
        V_network3_params[new_key] = v

# Initialize Lyapunov network with 3 UAVs
V_net = VectorLyapunovNetwork(input_dim=2*dim, hidden_dim=64, G=None, num_UAVs=4)

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
    state_dict[f'networks.0.{k}'] = v
for k, v in V_network2_params.items():
    state_dict[f'networks.1.{k}'] = v
for k, v in V_network3_params.items():
    state_dict[f'networks.2.{k}'] = v

# Check if we have all required keys
existing_keys = set(state_dict.keys())
required_keys = set(dict(V_net.named_parameters()).keys())
missing_keys = required_keys - existing_keys

'''
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
'''


V_net.load_state_dict(state_dict)


V_net.eval()

delta_ref = dynamics_params['desired_spacing'][0]  # X-direction spacing

# Create a grid for x-position error and x-velocity error
x_error_space = np.linspace(-3.0, 3.0, 100)  # X-position error range
vx_error_space = np.linspace(-3.0, 3.0, 100)  # X-velocity error range

# Initialize Lyapunov values for both followers
V1 = np.zeros((len(x_error_space), len(vx_error_space)))
V2 = np.zeros((len(x_error_space), len(vx_error_space)))
V3 = np.zeros((len(x_error_space), len(vx_error_space)))
# Compute Lyapunov values over the grid
for i, p_err in enumerate(x_error_space):
    for j, v_err in enumerate(vx_error_space):
        # Create a full 6D error state vector with zeros for y and z dimensions
        # Format: [px_err, py_err, pz_err, vx_err, vy_err, vz_err]
        x = torch.tensor([p_err, 0.0, 0.0, v_err, 0.0, 0.0, p_err, 0.0, 0.0, v_err, 0.0, 0.0, p_err, 0.0, 0.0, v_err, 0.0, 0.0], dtype=torch.float32)
        
        # Compute Lyapunov values for each follower
        with torch.no_grad():
            # For first follower (UAV 1)
            V = V_net(x)
            V1[i, j] = V[0]
            V2[i, j] = V[1]
            V3[i, j] = V[2]

# Create meshgrid for 3D plotting
X, Y = np.meshgrid(x_error_space, vx_error_space)
Z1 = V1.T  # Transpose to match meshgrid dimensions
Z2 = V2.T  # Transpose to match meshgrid dimensions
Z3 = V3.T  # Transpose to match meshgrid dimensions
# 3D Lyapunov plot for first follower
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

# 3D Lyapunov plot for third follower
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z3, cmap='viridis', antialiased=True)
ax.set_xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
ax.set_ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
ax.set_zlabel('Lyapunov Function (UAV 3)', fontsize=14, labelpad=10)
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.savefig('output_figures/uav3_lyapunov_3d.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# 2D contour plots
# First follower individual plot
fig = plt.figure(figsize=(10, 8), dpi=300)
contour1 = plt.contourf(X, Y, Z1, cmap='viridis', levels=20, alpha=0.95)
plt.plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
plt.xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
plt.ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
plt.title('Lyapunov Function (UAV 1)', fontsize=16)
plt.colorbar(contour1, label='Lyapunov Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output_figures/uav1_lyapunov_2d.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# Second follower individual plot
fig = plt.figure(figsize=(10, 8), dpi=300)
contour2 = plt.contourf(X, Y, Z2, cmap='plasma', levels=20, alpha=0.95)
plt.plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
plt.xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
plt.ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
plt.title('Lyapunov Function (UAV 2)', fontsize=16)
plt.colorbar(contour2, label='Lyapunov Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output_figures/uav2_lyapunov_2d.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# Third follower individual plot
fig = plt.figure(figsize=(10, 8), dpi=300)
contour3 = plt.contourf(X, Y, Z3, cmap='viridis', levels=20, alpha=0.95)
plt.plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
plt.xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
plt.ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
plt.title('Lyapunov Function (UAV 3)', fontsize=16)
plt.colorbar(contour3, label='Lyapunov Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output_figures/uav3_lyapunov_2d.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()


# Combined 2D contour plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

# Follower 1 contour
contour1 = axes[0].contourf(X, Y, Z1, cmap='viridis', levels=20, alpha=0.95)
axes[0].plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
axes[0].set_xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
axes[0].set_ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
axes[0].set_title('Lyapunov Function (UAV 1)', fontsize=16)
plt.colorbar(contour1, ax=axes[0], label='Lyapunov Value')
axes[0].legend()
axes[0].grid(True)

# Follower 2 contour
contour2 = axes[1].contourf(X, Y, Z2, cmap='plasma', levels=20, alpha=0.95)
axes[1].plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
axes[1].set_xlabel('X-Position Error (m)', fontsize=14, labelpad=10)
axes[1].set_ylabel('X-Velocity Error (m/s)', fontsize=14, labelpad=10)
axes[1].set_title('Lyapunov Function (UAV 2)', fontsize=16)
plt.colorbar(contour2, ax=axes[1], label='Lyapunov Value')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('output_figures/uav_lyapunov_contours.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()
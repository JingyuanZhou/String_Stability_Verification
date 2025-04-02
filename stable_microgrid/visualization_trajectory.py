import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
from networks import VectorLyapunovNetwork
from pre_train_model.learn_dynamics_control import DynamicsNN, ControllerNN
from networks import CombinedController
import os
from matplotlib import cm

# Microgrid Lyapunov Function Analysis
check_point = torch.load('model_weights/best_microgrid_model-v29.ckpt')
parameters = check_point['state_dict']

# Extract controller parameters for the CombinedController
controller_parameters = {}
controller1_params = {}
controller2_params = {}
controller3_params = {}
for k, v in parameters.items():
    if k.startswith('controller.controller_1'):
        new_key = k.replace('controller.controller_1.', '')
        controller1_params[new_key] = v
    if k.startswith('controller.controller_2'):
        new_key = k.replace('controller.controller_2.', '')
        controller2_params[new_key] = v
    if k.startswith('controller.controller_3'):
        new_key = k.replace('controller.controller_3.', '')
        controller3_params[new_key] = v

# Initialize Vector Lyapunov Network
state_dim = 3  # delta, omega, xi for each inverter
hidden_dim = 64
num_inverters = 3

# Create binary adjacency matrix for communication graph
G = torch.zeros(num_inverters, num_inverters)
for i in range(num_inverters):
    G[i, i] = 0.01  # Self-connection
    if i > 0:  # Connect to previous
        G[i, i-1] = G[i-1, i] = 0.01
    if i < num_inverters - 1:  # Connect to next
        G[i, i+1] = G[i+1, i] = 0.01

# Create CombinedController
combined_controller = CombinedController(input_dim=state_dim)
combined_controller.controller_1.load_state_dict(controller1_params)
combined_controller.controller_2.load_state_dict(controller2_params)
combined_controller.controller_3.load_state_dict(controller3_params)
combined_controller.eval()

# Extract Lyapunov network parameters
V_params = {}
V_network1_params = {}
V_network2_params = {}
V_network3_params = {}

for k, v in parameters.items():
    if k.startswith('V_net.network_1'):
        new_key = k.replace('V_net.network_1.', '')
        V_network1_params[new_key] = v
    if k.startswith('V_net.network_2'):
        new_key = k.replace('V_net.network_2.', '')
        V_network2_params[new_key] = v
    if k.startswith('V_net.network_3'):
        new_key = k.replace('V_net.network_3.', '')
        V_network3_params[new_key] = v

# Initialize Lyapunov network with 3 UAVs
V_net = VectorLyapunovNetwork(hidden_dim=64, G=None)
V_net.network_1.load_state_dict(V_network1_params)
V_net.network_2.load_state_dict(V_network2_params)
V_net.network_3.load_state_dict(V_network3_params)
# Define nominal frequency
omega_star = 2 * np.pi * 50  # 50 Hz

# Define parameter ranges for visualization
# delta_range = np.linspace(-5, 5, 100)  # Phase angle error range
delta_error_range = np.linspace(-100, 100, 100)  # Fixed value for controller state
omega_range = np.linspace(-10.0, 10.0, 100)  # Frequency error range
xi_fixed = 0.0  # Fixed value for controller state

# Create mesh grid for 3D plots
omega_mesh, delta_mesh = np.meshgrid(omega_range, delta_error_range)
v_values = np.zeros((3, len(omega_range), len(delta_error_range)))  # Storage for Lyapunov values
for i, omega in enumerate(omega_range):
    for j, delta in enumerate(delta_error_range):
        # Create state vector for all three inverters with the same error
        # (for simplicity, we're using the same error values for visualization)

        # Set state values for each inverter
        lyapunov_values = []
        for k in range(num_inverters):

            if k == 1:
                state = torch.zeros(1, state_dim + 2)
            else:
                state = torch.zeros(1, state_dim)

            if k == 0 or k == 2:
                state[0, 0] = delta     # delta error
                state[0, 1] = omega     # omega error
                state[0, 2] = xi_fixed  # xi (fixed)
            elif k == 1:
                state[0, 0] = delta     # delta error
                state[0, 1] = 0
                state[0, 2] = omega     # omega error'
                state[0, 3] = 0
                state[0, 4] = xi_fixed  # xi (fixed)

            if k == 0:
                lyapunov_value = V_net.network_1(state)
            elif k == 1:
                lyapunov_value = V_net.network_2(state)
            else:
                lyapunov_value = V_net.network_3(state)
            
            lyapunov_values.append(lyapunov_value)
            
        v_values[0, i, j] = lyapunov_values[0].item()
        v_values[1, i, j] = lyapunov_values[1].item()
        v_values[2, i, j] = lyapunov_values[2].item()

# Create separate 3D visualizations for each inverter's Lyapunov function
for inv_idx in range(num_inverters):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(omega_mesh, delta_mesh, v_values[inv_idx].T, 
                          cmap=cm.viridis, alpha=0.8, antialiased=True)
    
    # Add labels and title
    ax.set_xlabel('Frequency Error (rad/s)')
    ax.set_ylabel('Controller State (rad)')
    ax.set_zlabel('Lyapunov Value')
    ax.set_title(f'Inverter {inv_idx+1} Lyapunov Function')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(f'output_figures/microgrid_lyapunov_function_inverter{inv_idx+1}_3d.png', dpi=300)
    plt.close(fig)

# Create separate 2D contour plots for each inverter
for inv_idx in range(num_inverters):
    fig = plt.figure(figsize=(10, 8))
    contour = plt.contourf(omega_mesh, delta_mesh, v_values[inv_idx].T, 
                          levels=20, cmap=cm.viridis)
    plt.xlabel('Frequency Error (rad/s)')
    plt.ylabel('Controller State (rad)')
    plt.title(f'Inverter {inv_idx+1} Lyapunov Function')
    plt.grid(True)
    plt.colorbar(contour, label='Lyapunov Value')
    
    # Mark the equilibrium point
    plt.plot(0, 0, 'r*', markersize=10, label='Equilibrium')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'output_figures/microgrid_lyapunov_function_inverter{inv_idx+1}_2d.png', dpi=300)
    plt.close(fig)
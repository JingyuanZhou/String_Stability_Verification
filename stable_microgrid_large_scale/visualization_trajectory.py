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
import numpy as np

# Microgrid Lyapunov Function Analysis
check_point = torch.load('model_weights/best_microgrid_model-v229.ckpt')
parameters = check_point['state_dict']

# Extract Lyapunov network parameters
V_params = {}
V_network1_params = {}
V_network2_params = {}
num_inverters = 5
shared_lyapunov = True
state_dim = 3
print(parameters.keys())
for k, v in parameters.items():
    if k.startswith('V_net.networks.0'):
        new_key = k.replace('V_net.networks.0.', '')
        V_network1_params[new_key] = v
    if k.startswith('V_net.networks.1'):
        new_key = k.replace('V_net.networks.1.', '')
        V_network2_params[new_key] = v

# Initialize Lyapunov network with 3 UAVs
V_net = VectorLyapunovNetwork(hidden_dim=64, G=None, num_inverters=num_inverters, shared_lyapunov=shared_lyapunov)
V_net.networks[0].load_state_dict(V_network1_params)
V_net.networks[1].load_state_dict(V_network2_params)
# Define nominal frequency
omega_star = 2 * np.pi * 50  # 50 Hz

# Define parameter ranges for visualization
# delta_range = np.linspace(-5, 5, 100)  # Phase angle error range
delta_error_range = np.linspace(-np.pi/4, np.pi/4, 100)  # Fixed value for controller state -np.pi/4, np.pi/4
omega_range = np.linspace(-50.0, 50.0, 100)  # Frequency error range
xi_fixed = 0.0  # Fixed value for controller state

# Create mesh grid for 3D plots
omega_mesh, delta_mesh = np.meshgrid(omega_range, delta_error_range)
v_values = np.zeros((2, len(omega_range), len(delta_error_range)))  # Storage for Lyapunov values
for i, omega in enumerate(omega_range):
    for j, delta in enumerate(delta_error_range):
        # Create state vector for all three inverters with the same error
        # (for simplicity, we're using the same error values for visualization)

        # Set state values for each inverter
        lyapunov_values = []
        for k in range(2):

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

            equilibrium_state = torch.zeros_like(state)
            if k == 0:
                lyapunov_value = V_net.networks[0](state) - V_net.networks[0](equilibrium_state) + 0.001    
            elif k == 1:
                lyapunov_value = V_net.networks[1](state) - V_net.networks[1](equilibrium_state) + 0.001

            
            lyapunov_values.append(lyapunov_value)
            
        v_values[0, i, j] = lyapunov_values[0].item()
        v_values[1, i, j] = lyapunov_values[1].item()

# Create separate 3D visualizations for each inverter's Lyapunov function
for inv_idx in range(2):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(omega_mesh, delta_mesh, v_values[inv_idx].T, 
                          cmap=cm.viridis, alpha=0.8, antialiased=True)
    
    # Add labels and title
    ax.set_xlabel('Frequency Error (rad/s)')
    ax.set_ylabel('Delta Error (rad)')
    ax.set_zlabel('Lyapunov Value')
    ax.set_title(f'Inverter {inv_idx+1} Lyapunov Function')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(f'output_figures/microgrid_lyapunov_function_inverter{inv_idx+1}_3d.png', dpi=300)
    plt.close(fig)

# Create separate 2D contour plots for each inverter
for inv_idx in range(2):
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
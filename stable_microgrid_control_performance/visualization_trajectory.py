import matplotlib.pyplot as plt
import numpy as np
import torch
#from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
from networks import VectorLyapunovNetwork
from pre_train_model.learn_dynamics_control import DynamicsNN, ControllerNN
from networks import CombinedController
import os
from matplotlib import cm
import numpy as np
from networks import CombinedSystemDynamics

# Microgrid Lyapunov Function Analysis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
check_point = torch.load('model_weights/best_microgrid_model-v21.ckpt') #model_weights/best_microgrid_model-v19.ckpt best_microgrid_model_ISS-v1.ckpt
parameters = check_point['state_dict']

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24  # 设置默认字体大小
plt.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 24  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 24  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 24  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 24  # 图例字体大小

# Extract controller parameters for the CombinedController
controller_parameters = {}
original_controller_parameters = {}
controller1_params = {}
controller2_params = {}
controller3_params = {}
original_controller1_params = {}
original_controller2_params = {}
original_controller3_params = {}
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
    if k.startswith('original_controller.controller_1'):
        new_key = k.replace('original_controller.controller_1.', '')
        original_controller1_params[new_key] = v
    if k.startswith('original_controller.controller_2'):
        new_key = k.replace('original_controller.controller_2.', '')
        original_controller2_params[new_key] = v
    if k.startswith('original_controller.controller_3'):
        new_key = k.replace('original_controller.controller_3.', '')
        original_controller3_params[new_key] = v

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
combined_controller = CombinedController(input_dim=state_dim).to(device)
combined_controller.controller_1.load_state_dict(controller1_params)
combined_controller.controller_2.load_state_dict(controller2_params)
combined_controller.controller_3.load_state_dict(controller3_params)
combined_controller.eval()

original_combined_controller = CombinedController(input_dim=state_dim).to(device)
original_combined_controller.controller_1.load_state_dict(original_controller1_params)
original_combined_controller.controller_2.load_state_dict(original_controller2_params)
original_combined_controller.controller_3.load_state_dict(original_controller3_params)
original_combined_controller.eval()

# Visualize the trajectory of the system



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
V_net = VectorLyapunovNetwork(hidden_dim=64, G=None).to(device)
V_net.network_1.load_state_dict(V_network1_params)
V_net.network_2.load_state_dict(V_network2_params)
V_net.network_3.load_state_dict(V_network3_params)
# Define nominal frequency
omega_star = 2 * np.pi * 50  # 50 Hz

# Define parameter ranges for visualization
# delta_range = np.linspace(-5, 5, 100)  # Phase angle error range
delta_error_range = np.linspace(-np.pi/4, np.pi/4, 100)  # Fixed value for controller state -np.pi/4, np.pi/4
omega_range = np.linspace(-50.0, 50.0, 100)  # Frequency error range
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
                state = torch.zeros(1, state_dim + 2).to(device)
            else:
                state = torch.zeros(1, state_dim).to(device)

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

            equilibrium_state = torch.zeros_like(state).to(device)
            if k == 0:
                lyapunov_value = V_net.network_1(state) - V_net.network_1(equilibrium_state) + 0.001    
            elif k == 1:
                lyapunov_value = V_net.network_2(state) - V_net.network_2(equilibrium_state) + 0.001
            else:
                lyapunov_value = V_net.network_3(state) - V_net.network_3(equilibrium_state) + 0.001
            
            lyapunov_values.append(lyapunov_value)
            
        v_values[0, i, j] = lyapunov_values[0].item()
        v_values[1, i, j] = lyapunov_values[1].item()
        v_values[2, i, j] = lyapunov_values[2].item()

try:
    # Create separate 3D visualizations for each inverter's Lyapunov function
    for inv_idx in range(num_inverters):
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
except Exception as e:
    pass

# Create separate 2D contour plots for each inverter
for inv_idx in range(num_inverters):
    fig = plt.figure(figsize=(8, 6))
    contour = plt.contourf(omega_mesh, delta_mesh, v_values[inv_idx].T, 
                          levels=20, cmap=cm.viridis)
    plt.xlabel('Frequency Error (rad/s)', fontsize=26, labelpad=10)
    plt.ylabel('Controller State (rad)', fontsize=26, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=24)
    #plt.title(f'Inverter {inv_idx+1} Lyapunov Function')
    plt.grid(False)
    plt.colorbar(contour, label='Lyapunov Value')
    
    # Mark the equilibrium point
    plt.plot(0, 0, 'r*', markersize=12, label='Equilibrium', markeredgecolor='white', markeredgewidth=1)
    plt.legend(fontsize=24, frameon=True, fancybox=True, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(f'output_figures/microgrid_lyapunov_function_inverter{inv_idx+1}_2d.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def monte_carlo_single_step(controller, original_controller, num_runs=2000, dt=0.01, perturbation_range=0.3):
    """
    Perform Monte Carlo analysis with single-step evaluation for each random initial condition
    
    Args:
        controller: Optimized neural network controller
        original_controller: Original controller for comparison
        num_runs: Number of Monte Carlo runs
        dt: Time step size
        perturbation_range: Range for random initial conditions
        
    Returns:
        Dictionary with statistics from the Monte Carlo runs
    """
    # Load dynamics models
    dynamics = CombinedSystemDynamics(state_dim=3, neighbor_dim=3, control_dim=1).to(device)
    dynamics_models = [dynamics.dynamics_1, dynamics.dynamics_2, dynamics.dynamics_3]
    dynamics_models[0].load_state_dict(torch.load("pre_train_model/dynamics_model_0.pth"))
    dynamics_models[1].load_state_dict(torch.load("pre_train_model/dynamics_model_1.pth"))
    dynamics_models[2].load_state_dict(torch.load("pre_train_model/dynamics_model_2.pth"))
    
    # Set dynamics models to evaluation mode
    for model in dynamics_models:
        model.eval()
    
    # Statistics storage
    optimized_errors = []
    original_errors = []
    optimized_control_effort = []
    original_control_effort = []
    relative_improvement = []
    
    # Nominal frequency
    omega_star = 2 * np.pi * 50
    equilibrium_state = torch.tensor([[0.0, omega_star, 0.0]]).to(device)  # Equilibrium state for each inverter

    
    # Run multiple single-step evaluations
    for run in range(num_runs):
        if run % 500 == 0:
            print(f"Running Monte Carlo evaluation {run+1}/{num_runs}")
        
        # Generate random initial conditions
        np.random.seed(run)  # For reproducibility
        initial_states = []
        for i in range(3):  # Three inverters
            # Random perturbation with global + local components
            delta_perturb = (np.random.rand() - 0.5) * 10
            omega_perturb = (np.random.rand() - 0.5) *2
            xi_perturb = 0
            
            # Create initial state with perturbation
            initial_state = torch.tensor([[delta_perturb, omega_perturb, xi_perturb]]).to(device)
            
            initial_states.append(initial_state)
        
        # Take one step with optimized controller
        optimized_next_states = []
        for i in range(len(initial_states)):
            with torch.no_grad():
                # Get control input using appropriate controller
                if i == 0:
                    state_i = torch.cat([initial_states[i], initial_states[i+1]], dim=1)
                    u = controller.controller_1(state_i)
                elif i == 1:
                    state_i = torch.cat([initial_states[i], initial_states[i-1], initial_states[i+1]], dim=1)
                    u = controller.controller_2(state_i)
                else:
                    state_i = torch.cat([initial_states[i], initial_states[i-1]], dim=1)
                    u = controller.controller_3(state_i)
                
                # Apply one step of dynamics
                if i == 0:
                    state_i = initial_states[i] + equilibrium_state
                    neighbor_state = initial_states[i+1] + equilibrium_state
                    next_state = dynamics_models[0](state_i, neighbor_state, u)
                elif i == 1:
                    state_i = initial_states[i] + equilibrium_state
                    neighbor_state = torch.cat([initial_states[i-1] + equilibrium_state, initial_states[i+1] + equilibrium_state], dim=1)
                    next_state = dynamics_models[1](state_i, neighbor_state, u)
                else:
                    state_i = initial_states[i] + equilibrium_state
                    neighbor_state = initial_states[i-1] + equilibrium_state
                    next_state = dynamics_models[2](state_i, neighbor_state, u)

                optimized_next_states.append(next_state.detach().cpu().numpy())

        # Take one step with original controller
        original_next_states = []
        for i in range(len(initial_states)):
            with torch.no_grad():
                # Get control input using appropriate controller
                if i == 0:
                    state_i = torch.cat([initial_states[i], initial_states[i+1]], dim=1)
                    u = original_controller.controller_1(state_i)
                elif i == 1:
                    state_i = torch.cat([initial_states[i], initial_states[i-1], initial_states[i+1]], dim=1)
                    u = original_controller.controller_2(state_i)
                else:
                    state_i = torch.cat([initial_states[i], initial_states[i-1]], dim=1)
                    u = original_controller.controller_3(state_i)
                
                # Apply one step of dynamics
                if i == 0:
                    state_i = initial_states[i] + equilibrium_state
                    neighbor_state = initial_states[i+1] + equilibrium_state
                    next_state = dynamics_models[0](state_i, neighbor_state, u)
                elif i == 1:
                    state_i = initial_states[i] + equilibrium_state
                    neighbor_state = torch.cat([initial_states[i-1] + equilibrium_state, initial_states[i+1] + equilibrium_state], dim=1)
                    next_state = dynamics_models[1](state_i, neighbor_state, u)
                else:
                    state_i = initial_states[i] + equilibrium_state
                    neighbor_state = initial_states[i-1] + equilibrium_state
                    next_state = dynamics_models[2](state_i, neighbor_state, u)
                
                original_next_states.append(next_state.detach().cpu().numpy())
        
        # Calculate errors after one step (norm of state)
        optimized_error = 0
        original_error = 0
        for i in range(len(initial_states)):
            # Extract states (handle different state layouts)

            opt_delta = optimized_next_states[i][0, 0]
            opt_omega = optimized_next_states[i][0, 1]
            orig_delta = original_next_states[i][0, 0]
            orig_omega = original_next_states[i][0, 1]

            # Calculate error as norm
            opt_err = (opt_omega-omega_star)**2
            orig_err = (orig_omega-omega_star)**2
            
            optimized_error += opt_err
            original_error += orig_err
        
        # Average error across inverters
        optimized_error /= len(initial_states)
        original_error /= len(initial_states)

        optimized_error = np.sqrt(optimized_error)
        original_error = np.sqrt(original_error)
        
        # Store metrics
        optimized_errors.append(optimized_error)
        original_errors.append(original_error)
        
        # Calculate relative improvement
        if original_error > 0:
            rel_imp = (original_error - optimized_error) / original_error * 100
            relative_improvement.append(rel_imp)
    
    # Compile results
    results = {
        'optimized_errors': {
            'mean': np.mean(optimized_errors),
            'std': np.std(optimized_errors),
            'min': np.min(optimized_errors),
            'max': np.max(optimized_errors),
            'all': optimized_errors
        },
        'original_errors': {
            'mean': np.mean(original_errors),
            'std': np.std(original_errors),
            'min': np.min(original_errors),
            'max': np.max(original_errors),
            'all': original_errors
        },
        'relative_improvement': {
            'mean': np.mean(relative_improvement),
            'std': np.std(relative_improvement),
            'min': np.min(relative_improvement),
            'max': np.max(relative_improvement),
            'all': relative_improvement
        }
    }
    
    return results

# Run single-step Monte Carlo analysis
monte_carlo_results = monte_carlo_single_step(
    combined_controller,
    original_combined_controller,
    num_runs=500,
    dt=0.01,
    perturbation_range=0.3
)

# Print Monte Carlo results summary
print("\nSingle-Step Monte Carlo Analysis Results:")
print(f"Number of runs: 500")

print("\nState Error After One Step:")
print(f"  Optimized Controller: Mean={monte_carlo_results['optimized_errors']['mean']:.4f}, " +
      f"Std={monte_carlo_results['optimized_errors']['std']:.4f}")
print(f"  Original Controller: Mean={monte_carlo_results['original_errors']['mean']:.4f}, " +
      f"Std={monte_carlo_results['original_errors']['std']:.4f}")

print("\nRelative Error Improvement:")
print(f"  Mean={monte_carlo_results['relative_improvement']['mean']:.2f}%, " +
      f"Std={monte_carlo_results['relative_improvement']['std']:.2f}%")
print(f"  Min={monte_carlo_results['relative_improvement']['min']:.2f}%, " +
      f"Max={monte_carlo_results['relative_improvement']['max']:.2f}%")

# Create histogram of error improvement
plt.figure(figsize=(10, 6))
plt.hist(monte_carlo_results['relative_improvement']['all'], bins=30, alpha=0.7, color='blue')
plt.axvline(x=monte_carlo_results['relative_improvement']['mean'], color='red', linestyle='--', 
           label=f'Mean: {monte_carlo_results["relative_improvement"]["mean"]:.2f}%')
plt.xlabel('Error Reduction (%)')
plt.ylabel('Frequency')
plt.title('Performance Improvement Distribution (Single-Step Analysis)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('output_figures/monte_carlo_improvement_histogram.png', dpi=300)
plt.close()

# Create violin plots comparing errors
plt.figure(figsize=(10, 6))
error_data = [
    monte_carlo_results['optimized_errors']['all'],
    monte_carlo_results['original_errors']['all']
]
plt.violinplot(error_data, showmeans=True, showmedians=True)
plt.xticks([1, 2], ['Optimized Controller', 'Original Controller'])
plt.ylabel('State Error (After One Step)')
plt.title('Single-Step Error Comparison')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('output_figures/monte_carlo_error_violin.png', dpi=300)
plt.close()

import torch
import numpy as np
import matplotlib.pyplot as plt
from networks import VectorBarrierNetwork
import seaborn as sns

plt.style.use('seaborn-white')
plt.rcParams["font.family"] = "Times New Roman"

def visualize_barrier_function(barrier_net, vehicle_index=1, spacing_range=(5, 35), velocity_range=(0, 30), resolution=100):
    """
    Visualize the barrier function for a specific vehicle
    
    Args:
        barrier_net: Trained barrier network
        vehicle_index: Index of the vehicle to visualize (1 for first following vehicle)
        spacing_range: Range of spacing values to plot
        velocity_range: Range of velocity values to plot
        resolution: Number of points in each dimension
    """
    # Create meshgrid for spacing and velocity
    spacing = np.linspace(spacing_range[0], spacing_range[1], resolution)
    velocity = np.linspace(velocity_range[0], velocity_range[1], resolution)
    X, Y = np.meshgrid(spacing, velocity)

    # Initialize barrier values array
    Z = np.zeros_like(X)

    # Create equilibrium state
    eq_spacing = 20.0
    eq_velocity = 15.0
    num_vehicles = 4  # Assuming 4 vehicles as in the codebase

    # Evaluate barrier function across the grid
    for i in range(resolution):
        for j in range(resolution):
            # Create state vector for all vehicles
            state = torch.zeros(1, num_vehicles, 2)
            
            # Set equilibrium state for lead vehicle
            state[0, 0, 0] = eq_spacing
            state[0, 0, 1] = eq_velocity
            
            # Set test state for vehicle of interest
            state[0, vehicle_index, 0] = spacing[j]
            state[0, vehicle_index, 1] = velocity[i]
            
            # Set equilibrium states for other vehicles
            for k in range(num_vehicles):
                if k != 0 and k != vehicle_index:
                    state[0, k, 0] = eq_spacing
                    state[0, k, 1] = eq_velocity

            # Create reference state (equilibrium)
            x_star = torch.ones_like(state) * torch.tensor([eq_spacing, eq_velocity])
            
            # Evaluate barrier function
            with torch.no_grad():
                barrier_values = barrier_net(state, x_star)
                Z[i, j] = barrier_values[0, vehicle_index-1].item()

    # Create figure with higher resolution and better size ratio
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Create filled contour plot with a better colormap
    # Using 'RdYlBu_r' for a professional red-yellow-blue gradient
    levels = np.linspace(Z.min(), Z.max(), 30)  # More levels for smoother gradients
    contourf = plt.contourf(X, Y, Z, levels=levels, 
                           cmap='RdYlBu_r', alpha=0.8)
    
    # Add contour lines with better styling
    contour = plt.contour(X, Y, Z, levels=10, 
                         colors='k', linewidths=0.5, alpha=0.3)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Add colorbar with better styling
    cbar = plt.colorbar(contourf)
    cbar.set_label('Barrier Function Value', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    
    # Improve labels and title
    plt.xlabel('Inter-vehicle Spacing (m)', fontsize=10)
    plt.ylabel('Velocity (m/s)', fontsize=10)
    plt.title(f'Safety Barrier Function - Vehicle {vehicle_index}', 
             fontsize=12, pad=10)

    
    # Improve grid styling
    plt.grid(True, linestyle='--', alpha=0.2, color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure with better quality
    plt.savefig(f'output_figures/barrier_function_vehicle_{vehicle_index}.pdf', 
                format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load the trained barrier network
    state_dims = [2] * 4  # 4 vehicles with 2 states each
    barrier_net = VectorBarrierNetwork(state_dims)
    
    # Load checkpoint and extract V_net weights
    checkpoint = torch.load('model_weights/best_model-v108.ckpt')
    state_dict = checkpoint['state_dict']
    
    # Print all keys to see what's available
    print("Available keys in state_dict:")
    for key in state_dict.keys():
        print(key)
    
    # Create new state dict with correct keys
    barrier_net_state_dict = {}
    for key, value in state_dict.items():
        # Print keys that start with barrier_net
        if key.startswith('barrier_net.'):
            new_key = key[12:]  # Skip 'barrier_net.'
            barrier_net_state_dict[new_key] = value
    
    
    # Load the modified state dict
    barrier_net.load_state_dict(barrier_net_state_dict)
    barrier_net.eval()

    # Visualize barrier function for vehicles 1, 2, and 3
    for vehicle_index in [1, 2, 3]:
        visualize_barrier_function(barrier_net, vehicle_index)

if __name__ == "__main__":
    main() 
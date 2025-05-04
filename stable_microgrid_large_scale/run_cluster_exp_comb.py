from training_exp_comb import train_model, retrain_model
from pre_train_model.learn_dynamics_control import ControllerNN, DynamicsNN
import torch
import os
from datetime import datetime
from generate_combined_model_torch_comb import combined_model
from queries_comb import decentralized_verification
import warnings
import numpy as np
from networks import CombinedSystemDynamics
warnings.filterwarnings("ignore")

# System parameters
num_inverters = 10  # Number of inverters in the microgrid
controlled_indices = list(range(1, num_inverters))  # Control all inverters except the first one
state_dims = [3] * num_inverters  # Each inverter has (delta, omega, xi)
control_dims = [1] * num_inverters  # Each inverter has one control input
num_ce_list = []
num_veri_time_list = []
num_training_time_list = []
num_total_time_list = []
shared_lyapunov = True
# Training parameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 100
max_iters = 100

# File paths
index = 0
out_comb_folders = "combined/"
cur_comb_file = out_comb_folders + f"combined_{index}.onnx"
pre_trained_model = f"pre_train_model/controller_model.pth"
pre_trained_dynamics = f"pre_train_model/dynamics_model.pth"

combined_num_inverters = num_inverters
if shared_lyapunov:
    combined_num_inverters = 6
    controlled_indices = list(range(1, combined_num_inverters))  # Control all inverters except the first one
    state_dims = [3] * combined_num_inverters  # Each inverter has (delta, omega, xi)
    control_dims = [1] * combined_num_inverters  # Each inverter has one control input

st_total_time = datetime.now()
combined_system_dynamics = CombinedSystemDynamics(state_dim=state_dims[0], neighbor_dim=state_dims[0], control_dim=control_dims[0], hidden_dim=64,num_inverters=combined_num_inverters, shared_lyapunov=shared_lyapunov)

# Dynamics parameters for microgrid
def create_line_topology_B(n, susceptance_value=0.01):
    """Create a susceptance matrix for a line topology with n inverters"""
    B = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            B[i, i-1] = susceptance_value
        if i < n-1:
            B[i, i+1] = susceptance_value
    return B

dynamics_params = {
    'dt': 0.01,  # Time step (s)
    'n': combined_num_inverters,  # Number of inverters
    'omega_star': 2 * np.pi * 50,  # Nominal frequency (50 Hz)
    'tau': [1.4895] * combined_num_inverters,  # Time constants
    'eta': [6.3509e-4] * combined_num_inverters,  # Droop gains
    'k': [4.9481] * combined_num_inverters,  # Secondary control gains
    'V': [325.3] * combined_num_inverters,  # Voltage magnitudes
    'B': create_line_topology_B(combined_num_inverters),  # Line susceptances (dynamically created)
    'P_L': [1260.0] * combined_num_inverters,  # Load powers
    'P_star': [1260.0] * combined_num_inverters  # Desired power injections
}

# Initial training
st_train_time = datetime.now()
controller, system, V_net = train_model(
    num_inverters=combined_num_inverters,
    controlled_indices=controlled_indices,
    state_dims=state_dims,
    control_dims=control_dims,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=num_epochs,
    dynamics_params=dynamics_params,
    shared_lyapunov=shared_lyapunov
)
end_train_time = datetime.now()
diff = end_train_time - st_train_time
num_training_time_list.append(diff.seconds)
print("Total training time for model index", str(index), ":", str(diff.seconds))

# Convert and combine models
combined_model(V_net, controller, combined_system_dynamics, cur_comb_file, state_dims, controlled_indices, shared_lyapunov)

# Verification 
st_ver_time = datetime.now()
ret, ret_ranges, failed = decentralized_verification(
    cur_comb_file,
    system=system, 
    num_inverters=combined_num_inverters,
    shared_lyapunov=shared_lyapunov
)

end_ver_time = datetime.now()
diff_ver_time = end_ver_time - st_ver_time
print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds))
print("Total counter examples found:", str(len(ret)) + "\n")
num_ce_list.append(len(ret))
num_veri_time_list.append(diff_ver_time.seconds)

# Retraining loop
while (len(ret) > 0) and (index < max_iters):
    index += 1
    learning_rate = learning_rate * 0.9
    st_train_time = datetime.now()
    controller, system, V_net = retrain_model(system, ret, ret_ranges, num_epochs, V_net, controller, learning_rate, batch_size, index, shared_lyapunov, combined_num_inverters)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    num_training_time_list.append(diff.seconds)
    print("Total training time for model index", str(index), ":", str(diff.seconds))

    combined_model(V_net, controller, combined_system_dynamics, cur_comb_file, state_dims, controlled_indices, shared_lyapunov)

    st_ver_time = datetime.now()
    ret, ret_ranges, failed = decentralized_verification(
        cur_comb_file, 
        system=system,
        num_inverters=combined_num_inverters,
        ret_ranges=ret_ranges,
        shared_lyapunov=shared_lyapunov
    )
    end_ver_time = datetime.now()
    diff_ver_time = end_ver_time - st_ver_time
    print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds) + "\n")
    print("Total counter examples found:", str(len(ret)) + "\n")
    num_ce_list.append(len(ret))
    num_veri_time_list.append(diff_ver_time.seconds)

# Save results
end_total_time = datetime.now()
diff_total_time = end_total_time - st_total_time
num_total_time_list.append(diff_total_time.seconds)
print("Total time for all iterations:", str(diff_total_time.seconds))

if shared_lyapunov:
    np.save("data/num_ce_list_" + str(num_inverters) + "_shared.npy", num_ce_list)
    np.save("data/num_veri_time_list_" + str(num_inverters) + "_shared.npy", num_veri_time_list)
    np.save("data/num_training_time_list_" + str(num_inverters) + "_shared.npy", num_training_time_list)
    np.save("data/num_total_time_list_" + str(num_inverters) + "_shared.npy", num_total_time_list)
else:
    np.save("data/num_ce_list_" + str(num_inverters) + ".npy", num_ce_list)
    np.save("data/num_veri_time_list_" + str(num_inverters) + ".npy", num_veri_time_list)
    np.save("data/num_training_time_list_" + str(num_inverters) + ".npy", num_training_time_list)
    np.save("data/num_total_time_list_" + str(num_inverters) + ".npy", num_total_time_list)
print(failed)

# Print final summary
print("\nTraining Summary:")
print(f"Number of inverters: {num_inverters}")
print(f"Controlled inverters: {controlled_indices}")
print(f"Nominal frequency: {dynamics_params['omega_star']/(2*np.pi):.2f} Hz")
print(f"Time step: {dynamics_params['dt']} s")
print(f"Training epochs: {num_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Total iterations: {index + 1}")
print(f"Final counter examples: {len(ret)}")


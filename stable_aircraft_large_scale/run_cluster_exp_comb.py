from training_exp_comb import train_model, retrain_model
from pre_train_model.learn_dynamics_control import ControllerNN, DynamicsNN
import torch
import os
from datetime import datetime
from generate_combined_model_torch_comb import combined_model
from queries_comb import centralized_verification, decentralized_verification
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# System parameters
num_UAVs = 6
controlled_indices = list(range(1, num_UAVs)) 
state_dims = [6] * num_UAVs
control_dims = [3] * num_UAVs
num_ce_list = []
num_veri_time_list = []
num_training_time_list = []
num_total_time_list = []
shared_Lyapunov = False

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
# Convert and combine models
if shared_Lyapunov:
    combined_state_dims = [6] * 4
    combined_num_UAVs = 4
else:
    combined_state_dims = state_dims
    combined_num_UAVs = num_UAVs
# Dynamics parameters
dynamics_params = {
    'dt': 0.1,
    'dim': 3,
    'max_thrust': 5.0,
    'min_thrust': -5.0,
    'cruise_velocity': np.array([5.0, 0.0, 0.0]),  # Mainly flying along X-axis
    'delta_ref': np.array([10.0, 0.0, 0.0]),  # Spacing along X-axis
    'dist_amplitude': np.array([0.2, 0.5, 0.3]),  # Disturbance amplitude
    'dist_frequency': np.array([0.5, 0.3, 0.4])  # Disturbance frequency
}
st_total_time = datetime.now()
st_train_time = datetime.now()
controller, system, V_net = train_model(
    num_uavs=combined_num_UAVs,
    controlled_indices=controlled_indices,
    state_dims=combined_state_dims,
    control_dims=control_dims,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=num_epochs,
    dynamics_params=dynamics_params,
    shared_Lyapunov=shared_Lyapunov
)
end_train_time = datetime.now()
diff = end_train_time - st_train_time
num_training_time_list.append(diff.seconds)
print("Total training time for model index", str(index), ":", str(diff.seconds))


combined_model(V_net, controller, system.neural_system, cur_comb_file, combined_state_dims, controlled_indices)

# Verification 
st_ver_time = datetime.now()
ret, ret_ranges, failed = decentralized_verification(
    cur_comb_file,
    system=system, 
    num_uavs= combined_num_UAVs,
    shared_Lyapunov=shared_Lyapunov
)

end_ver_time = datetime.now()
diff_ver_time = end_ver_time - st_ver_time
print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds))
print("Total counter examples found:", str(len(ret)) + "\n")
num_ce_list.append(len(ret))
num_veri_time_list.append(diff_ver_time.seconds)

while (len(ret) > 0) and (index < max_iters):
    index += 1
    learning_rate = learning_rate * 0.9
    st_train_time = datetime.now()
    controllers, system, V_net = retrain_model(system, ret, ret_ranges, num_epochs, V_net, controller, learning_rate, batch_size, index, shared_Lyapunov=shared_Lyapunov, num_uavs=combined_num_UAVs)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    num_training_time_list.append(diff.seconds)
    print("Total training time for model index", str(index), ":", str(diff.seconds))

    combined_model(V_net, controllers, system.neural_system, cur_comb_file, combined_state_dims, controlled_indices)

    st_ver_time = datetime.now()
    ret, ret_ranges, failed = decentralized_verification(
        cur_comb_file, 
        system=system,
        num_uavs= combined_num_UAVs,
        ret_ranges = ret_ranges,
        shared_Lyapunov=shared_Lyapunov
    )
    #print(ret_ranges)
    end_ver_time = datetime.now()
    diff_ver_time = end_ver_time - st_ver_time
    print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds) + "\n")
    print("Total counter examples found:", str(len(ret)) + "\n")
    num_ce_list.append(len(ret))
    num_veri_time_list.append(diff_ver_time.seconds)

end_total_time = datetime.now()
diff_total_time = end_total_time - st_total_time
num_total_time_list.append(diff_total_time.seconds)
print("Total time for total experiment:", str(diff_total_time.seconds))
if shared_Lyapunov:
    np.save("data/num_ce_list_"+ str(num_UAVs) + "_shared.npy", num_ce_list)
    np.save("data/num_veri_time_list_"+ str(num_UAVs) + "_shared.npy", num_veri_time_list)
    np.save("data/num_training_time_list_"+ str(num_UAVs) + "_shared.npy", num_training_time_list)
    np.save("data/num_total_time_list_"+ str(num_UAVs) + "_shared.npy", num_total_time_list)
else:
    np.save("data/num_ce_list_"+ str(num_UAVs) + ".npy", num_ce_list)
    np.save("data/num_veri_time_list_"+ str(num_UAVs) + ".npy", num_veri_time_list)
    np.save("data/num_training_time_list_"+ str(num_UAVs) + ".npy", num_training_time_list)
    np.save("data/num_total_time_list_"+ str(num_UAVs) + ".npy", num_total_time_list)
print(failed)


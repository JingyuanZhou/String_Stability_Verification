from training_exp_comb import train_model, VectorLyapunovNetwork
import torch
import os
from datetime import datetime
from convertsinglenetwork import single_model
from generate_combined_model_torch_comb import combined_model, combine_prev_cur
# from queries_comb import safe_descent_cond_check

# Create output directories
out_folders = ["controllers/", "models/", "data/", "combined/", "counterexamples/", "model_weights/"]
for folder in out_folders:
    if not os.path.isdir(folder):
        os.mkdir(folder)

# System parameters
num_vehicles = 3
cav_indices = [1]  # Second vehicle is CAV
state_dims = [2] * num_vehicles
control_dims = [1] * num_vehicles

# Dynamics parameters
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

# Training parameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

# File paths
index = 0
out_comb_folders = "combined/"
cur_comb_file = out_comb_folders + f"combined_{index}.onnx"
cur_models_file = out_comb_folders + f"models_{index}.onnx"
cur_model_onnx_file = f"models/cert_{index}.onnx"

# Initialize and save empty V_net as previous model
prev_V_net = VectorLyapunovNetwork(state_dims)

# Train the model
st_train_time = datetime.now()
controllers, system, V_net = train_model(
    num_vehicles=num_vehicles,
    cav_indices=cav_indices,
    state_dims=state_dims,
    control_dims=control_dims,
    dynamics_params=dynamics_params,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=num_epochs
)
end_train_time = datetime.now()
diff = end_train_time - st_train_time
print("Total training time for model index", str(index), ":", str(diff.seconds))

# Convert and combine models
combined_model(V_net, controllers, cur_comb_file, state_dims, cav_indices)
#combine_prev_cur(V_net, cur_models_file, state_dims)
#single_model(V_net, cur_model_onnx_file, state_dims)


'''
# Verification
st_ver_time = datetime.now()
ret, ret_ranges, failed = safe_descent_cond_check(
    cur_comb_file, 
    cur_model_onnx_file, 
    cur_models_file, 
    prev_pos=prev_pos_bound, 
    safe_pos=4.1, 
    limit_pos=5, 
    docking_pos=0.35, 
    vel_limit=0.5
)
end_ver_time = datetime.now()
diff_ver_time = end_ver_time - st_ver_time
print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds))

while (len(ret) > 0):
    index += 1

    st_train_time = datetime.now()
    retrain_model(torch.Tensor(ret), torch.Tensor(ret_ranges), 4.1, 5, 0.5, cur_train_file, next_train_file, next_val_file, cur_model_file, cur_controller_file, next_model_file, next_controller_file, threshold, prev_model_file, prev_pos_bound)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    print("Total training time for model index", str(index), ":", str(diff.seconds))

    combined_model(next_model_file, next_controller_file, next_comb_file)
    combine_prev_cur(next_model_file, prev_model_file, next_models_file)
    single_model(next_model_file, next_model_onnx_file)

    st_ver_time = datetime.now()
    ret, ret_ranges, failed = safe_descent_cond_check(next_comb_file, next_model_onnx_file, next_models_file, prev_pos = prev_pos_bound, safe_pos = 4.1, limit_pos = 5, docking_pos = 0.35, vel_limit = 0.5)
    end_ver_time = datetime.now()
    diff_ver_time = end_ver_time - st_ver_time
    print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds) + "\n")
    cur_train_file, cur_val_file, cur_model_file, cur_model_onnx_file, cur_controller_file, cur_models_file = next_train_file, next_val_file, next_model_file, next_controller_file, next_models_file

print(failed)
'''
from training_exp_comb import train_model, retrain_model, system_network
import torch
import os
from datetime import datetime
from generate_combined_model_torch_comb import combined_model
from queries_comb import safe_descent_cond_check

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
    'a_max': 7.0,
    'a_min': -7.0,
    'desired_spacing': 20.0
}

# Training parameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 15

max_iters = 10000

# File paths
index = 0
out_comb_folders = "combined/"
cur_comb_file = out_comb_folders + f"combined_{index}.onnx"
pre_trained_model = "pre_train_model/sac_platoon_-1_actor.pth"

# Train the model
system_dynamics_network = system_network(state_dim=3)

st_train_time = datetime.now()
controllers, system, V_net = train_model(
    num_vehicles=num_vehicles,
    cav_indices=cav_indices,
    state_dims=state_dims,
    control_dims=control_dims,
    dynamics_params=dynamics_params,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=num_epochs,
    system_dynamics_network=system_dynamics_network,
    train_system=False,
    index=index,
    pre_trained_model=pre_trained_model
)
end_train_time = datetime.now()
diff = end_train_time - st_train_time
print("Total training time for model index", str(index), ":", str(diff.seconds))

# Convert and combine models
combined_model(V_net, controllers[1], system_dynamics_network, cur_comb_file, state_dims, cav_indices)

# Verification
st_ver_time = datetime.now()
ret, ret_ranges, failed = safe_descent_cond_check(
    cur_comb_file,
    system=system, 
    num_agents= 3
)
end_ver_time = datetime.now()
diff_ver_time = end_ver_time - st_ver_time
print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds))
print("Total counter examples found:", str(len(ret)) + "\n")

while (len(ret) > 0) and (index < max_iters):
    index += 1

    st_train_time = datetime.now()
    controllers, system, V_net = retrain_model(num_vehicles=num_vehicles, cav_indices=cav_indices, state_dims=state_dims, 
                                               control_dims=control_dims, dynamics_params=dynamics_params, counterexamples=torch.Tensor(ret), 
                                               counterexample_ranges=ret_ranges, epoch=num_epochs, in_model= V_net, 
                                               in_controller = controllers, index = index)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    print("Total training time for model index", str(index), ":", str(diff.seconds))

    combined_model(V_net, controllers[1], system_dynamics_network, cur_comb_file, state_dims, cav_indices)

    st_ver_time = datetime.now()
    ret, ret_ranges, failed = safe_descent_cond_check(
        cur_comb_file, 
        system=system,
        num_agents= 3
    )
    end_ver_time = datetime.now()
    diff_ver_time = end_ver_time - st_ver_time
    print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds) + "\n")
    print("Total counter examples found:", str(len(ret)) + "\n")

print(failed)


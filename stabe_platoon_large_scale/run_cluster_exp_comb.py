from training_exp_comb import train_model, retrain_model, system_network
import torch
import os
from datetime import datetime
from generate_combined_model_torch_comb import combined_model
from queries_comb import centralized_verification, decentralized_verification
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# System parameters
num_vehicles = 50
shared_Lyapunov = False
cav_indices = [1]  # Second vehicle is CAV
state_dims = [2] * num_vehicles
control_dims = [1] * num_vehicles
num_ce_list = []
num_veri_time_list = []
num_training_time_list = []
total_time = 0

# Dynamics parameters
dynamics_params = {
    'dt': 0.1,
    'alpha': 0.6,
    'beta': 0.9,
    'v_max': 30.0,
    's_st': 5.0,
    's_go': 35.0,
    'a_max': 100.0,
    'a_min': -100.0,
    'desired_spacing': 20.0
}

# Training parameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 100

max_iters = 100

# File paths
index = 0
out_comb_folders = "combined/"
cur_comb_file = out_comb_folders + f"combined_{index}.onnx"
pre_trained_id = 95
pre_trained_model = f"pre_train_model/sac_platoon_{pre_trained_id}_actor.pth"
pre_trained_critics = f"pre_train_model/sac_platoon_{pre_trained_id}_critic.pth"
#pre_trained_model = None
#pre_trained_critics = None
st_total_time = datetime.now()
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
    pre_trained_model=pre_trained_model,
    pre_trained_critics = pre_trained_critics,
    shared_Lyapunov = shared_Lyapunov
)
end_train_time = datetime.now()
diff = end_train_time - st_train_time
print("Total training time for model index", str(index), ":", str(diff.seconds))
num_training_time_list.append(diff.seconds)

# Convert and combine models
combined_model(V_net, controllers, system_dynamics_network, cur_comb_file, state_dims, cav_indices)

# Verification 
st_ver_time = datetime.now()
ret, ret_ranges, failed = decentralized_verification(
    cur_comb_file,
    system=system, 
    num_agents= num_vehicles,
    shared_Lyapunov = shared_Lyapunov
)
#print(ret_ranges)
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
    controllers, system, V_net = retrain_model(num_vehicles=num_vehicles, cav_indices=cav_indices, state_dims=state_dims, 
                                               control_dims=control_dims, in_system=system, counterexamples=torch.Tensor(ret), 
                                               counterexample_ranges=ret_ranges, epoch=num_epochs, in_model= V_net, learning_rate=learning_rate,
                                               in_controller = controllers, index = index, pre_trained_model=pre_trained_model, pre_trained_critics = pre_trained_critics, combined_model_path=cur_comb_file)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    print("Total training time for model index", str(index), ":", str(diff.seconds))
    num_training_time_list.append(diff.seconds)
    combined_model(V_net, controllers, system_dynamics_network, cur_comb_file, state_dims, cav_indices)

    st_ver_time = datetime.now()
    ret, ret_ranges, failed = decentralized_verification(
        cur_comb_file, 
        system=system,
        num_agents= num_vehicles,
        ret_ranges = ret_ranges,
        shared_Lyapunov = shared_Lyapunov
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
print("Total time for all experiments:", str(diff_total_time.seconds))
total_time = diff_total_time.seconds

if shared_Lyapunov:
    np.save("data/num_ce_list_shared_"+str(num_vehicles)+".npy", num_ce_list)
    np.save("data/num_veri_time_list_shared_"+str(num_vehicles)+".npy", num_veri_time_list)
    np.save("data/num_training_time_list_shared_"+str(num_vehicles)+".npy", num_training_time_list)
    np.save("data/total_time_shared_"+str(num_vehicles)+".npy", total_time)
else:
    np.save("data/num_ce_list_"+str(num_vehicles)+".npy", num_ce_list)
    np.save("data/num_veri_time_list_"+str(num_vehicles)+".npy", num_veri_time_list)
    np.save("data/num_training_time_list_"+str(num_vehicles)+".npy", num_training_time_list)
    np.save("data/total_time_"+str(num_vehicles)+".npy", total_time)
print(failed)


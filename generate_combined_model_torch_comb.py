import torch
import torch.nn as nn
import torch.onnx
from onnxsim import simplify
import onnx

def combined_model(V_net, controllers, system_dynamics, output_file, state_dims, cav_indices): 
    """
    Combine V_net, controllers and previous model into a single ONNX model
    
    Args:
        V_net: Vector Lyapunov network
        controllers: ModuleList of controllers
        prev_V_net: Previous Vector Lyapunov network
        output_file: Path to save combined ONNX model
        state_dims: List of state dimensions for each vehicle
        cav_indices: List of CAV indices
    """
    class CombinedNetwork(nn.Module):
        def __init__(self, controllers, V_net, cav_indices, state_dims, system_dynamics):
            super(CombinedNetwork, self).__init__()
            self.controllers = controllers
            self.controllers_temp = controllers
            self.V_net_1 = V_net
            self.V_net_2 = V_net
            self.system_dynamics = system_dynamics
            self.cav_indices = cav_indices
            self.state_dims = state_dims
            self.batch_size = 1
            W_state_with_preceding = torch.zeros(6, 3)
            W_state_with_preceding[3:6, torch.arange(3)] = 1.0
            self.register_buffer('W_state_with_preceding', W_state_with_preceding)

            '''
            W_leading_vehicle = torch.zeros(6, 2)
            W_CAV = torch.zeros(6, 2)
            W_HDV = torch.zeros(6, 2)

            W_leading_vehicle[0:2, torch.arange(2)] = 1.0
            W_CAV[2:4, torch.arange(2)] = 1.0
            W_HDV[4:6, torch.arange(2)] = 1.0

            self.register_buffer('W_leading_vehicle', W_leading_vehicle)
            self.register_buffer('W_CAV', W_CAV)
            self.register_buffer('W_HDV', W_HDV)
            '''
            self.dt = 0.1
            self.A = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, self.dt, 1, -self.dt, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, self.dt, 1, -self.dt], [0, 0, 0, 0, 0, 1]],dtype=torch.float32)
            self.b = torch.tensor([[0,0,0],[0,0,0],[0,0,0],[0,self.dt,0],[0,0,0],[0,0,self.dt]],dtype=torch.float32)
            self.A = self.A.unsqueeze(0)
            self.b = self.b.unsqueeze(0)
            # Use a simple Euler update for each vehicle.
            # x shape: [batch_size, num_vehicles, 2]
            # u shape: [batch_size, number_of_CAVs]
            # a shape: [batch_size, number_of_vehicles] (or similar)

   
        def forward(self, x):
            # x, y shape: [batch_size, num_vehicles * 2]
            batch_size = 1#x.shape[0]
            num_vehicles = len(self.state_dims)
            
            # 创建期望状态向量 [batch_size, num_vehicles, 2]
            x_stars = torch.tensor([[20.0, 15.0]] * num_vehicles, device=x.device)  # [num_vehicles, 2]
            x_stars = x_stars.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_vehicles, 2]
            
            # 只输出CAV的控制器输出
            u_bounds = (torch.tensor(-5.0, device=x.device), 
                          torch.tensor(5.0, device=x.device))
            u_star = torch.zeros(1, device=x.device)
            output_controllers = self.controllers(x, x_stars, u_star, u_bounds)

            x_with_preceding = torch.matmul(x.view(batch_size, -1), self.W_state_with_preceding)  # [batch_size, 6]
            a_HDV = self.system_dynamics(x_with_preceding)  # 根据系统动力学定义
            
            controllers_temp = self.controllers_temp(x, x_stars, u_star, u_bounds)
            zero_tensor = controllers_temp * 0
            acceleration = torch.cat([zero_tensor,controllers_temp,a_HDV], dim=1)  # [batch_size, num_vehicles]
        
            next_state = torch.matmul(self.A, x.reshape(2*num_vehicles)) + torch.matmul(self.b, acceleration.reshape(num_vehicles))
            output_V = self.V_net_1(x, x_stars)
            next_V = self.V_net_2(next_state, x_stars)

            return output_V, next_state, next_V
    
    # Create and export combined model
    combined_network = CombinedNetwork(controllers, V_net, cav_indices, state_dims, system_dynamics)
    
    # 创建包含所有车辆状态的dummy输入
    #dummy_input_x = torch.randn(1, len(state_dims), state_dims[0],requires_grad=True)  # [1, num_vehicles]
    dummy_input_x = torch.tensor([[[20,15],[21,14],[21,13]]],requires_grad=True, dtype=torch.float32)  # [1, num_vehicles]
    
    #print("dummy_input_x", dummy_input_x)
    #output1, output2, output3 = combined_network(dummy_input_x)
    output1 = combined_network(dummy_input_x)
    #print("input_x", dummy_input_x)
    #print("output1", output1)
    #print("output2", output2)
    #print("output3", output3)

    torch.onnx.export(
        combined_network,
        (dummy_input_x),
        output_file,
        export_params=True,opset_version=10,do_constant_folding=True,
        input_names=['input_x'],
        output_names=['output_V','next_state','next_V'],
    )

    model = onnx.load(output_file)
    model_simp, check = simplify(model)
    onnx.save(model_simp, output_file)

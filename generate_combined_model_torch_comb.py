import torch
import torch.nn as nn
import torch.onnx

from attempt_conversion import LearnedController

def combined_model(V_net, controllers, output_file, state_dims, cav_indices): 
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
        def __init__(self, controllers, V_net, cav_indices, state_dims):
            super(CombinedNetwork, self).__init__()
            self.controllers = controllers
            self.V_net = V_net
            self.cav_indices = cav_indices
            self.state_dims = state_dims
        
        def forward(self, x, y):
            # x, y shape: [batch_size, num_vehicles * 2]
            batch_size = x.shape[0]
            num_vehicles = self.state_dims[1]
            
            # 创建期望状态向量 [batch_size, num_vehicles, 2]
            x_stars = torch.tensor([[20.0, 15.0]] * num_vehicles, device=x.device)  # [num_vehicles, 2]
            x_stars = x_stars.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_vehicles, 2]
            
            # 只输出CAV的控制器输出
            output_controllers = []
            for i in self.cav_indices:
                state_i = x[:, i, :]  # 获取第i辆车的状态
                x_star_i = x_stars[:, i, :]  # 获取第i辆车的期望状态
                u_star = torch.zeros(1, device=x.device)
                u_bounds = (torch.tensor(-5.0, device=x.device), 
                          torch.tensor(5.0, device=x.device))
                control = self.controllers[i](state_i, x_star_i, u_star, u_bounds)
                output_controllers.append(control)
            output_controllers = torch.cat(output_controllers, dim=-1)
            
            output_V1 = self.V_net(x, x_stars)
            output_V2 = self.V_net(y, x_stars)
            
            return output_controllers, output_V1, output_V2
    
    # Create and export combined model
    combined_network = CombinedNetwork(controllers, V_net, cav_indices, state_dims)
    
    # 创建包含所有车辆状态的dummy输入
    dummy_input_x = torch.randn(1, state_dims[1], state_dims[0])  # [1, num_vehicles * 2]
    dummy_input_y = torch.randn(1, state_dims[1], state_dims[0])  # [1, num_vehicles * 2]
    
    torch.onnx.export(
        combined_network,
        (dummy_input_x, dummy_input_y),
        output_file,
        input_names=['input_x', 'input_y'],
        output_names=['controllers_out', 'V1_out', 'V2_out'],
        dynamic_axes={'input_x': {0: 'batch_size'},
                     'input_y': {0: 'batch_size'}}
    )

def combine_prev_cur(V_net, output_file, state_dims):
    """
    Combine current and previous V_net into a single ONNX model
    
    Args:
        V_net: Current Vector Lyapunov network
        prev_V_net: Previous Vector Lyapunov network
        output_file: Path to save combined ONNX model
        state_dims: List of state dimensions for each vehicle
    """
    class CombinedNetwork(nn.Module):
        def __init__(self, V_net, state_dims):
            super(CombinedNetwork, self).__init__()
            self.V_net = V_net
            self.state_dims = state_dims

        def forward(self, x):
            batch_size = x.shape[0]
            num_vehicles = self.state_dims[1]
            x_stars = torch.tensor([[20.0, 15.0]] * num_vehicles, device=x.device)  # [num_vehicles, 2]
            x_stars = x_stars.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_vehicles, 2]
            
            output_current = self.V_net(x, x_stars)

            return output_current

    # Create and export combined model
    combined_network = CombinedNetwork(V_net, state_dims)
    

    dummy_input = torch.randn(1, state_dims[1], state_dims[0])  
    
    torch.onnx.export(
        combined_network,
        dummy_input,
        output_file,
        input_names=['input'],
        output_names=['output_current'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
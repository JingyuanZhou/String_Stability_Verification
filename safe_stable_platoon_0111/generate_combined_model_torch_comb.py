import torch
import torch.nn as nn
import torch.onnx
from onnxsim import simplify
import onnx
from networks import CombinedControllers

class CombinedNetwork(nn.Module):
    def __init__(self, controllers, V_net, B_net, cav_indices, state_dims, system_dynamics):
        super(CombinedNetwork, self).__init__()
        self.controllers = controllers
        self.CombindedControllers = CombinedControllers(controllers)
        self.V_net = V_net
        self.B_net = B_net

        self.system_dynamics = system_dynamics
        self.cav_indices = cav_indices
        self.state_dims = state_dims
        self.batch_size = 1
        self.num_vehicles = len(state_dims)
        
        # 生成HDV索引列表
        self.hdv_indices = [i for i in range(1, self.num_vehicles) if i not in cav_indices]

        # CAV选择矩阵
        W_cav = torch.zeros(self.num_vehicles, len(cav_indices))
        for i, cav_idx in enumerate(cav_indices):
            W_cav[cav_idx, i] = 1.0
        self.register_buffer('W_cav', W_cav)

        # HDV选择矩阵
        W_hdv = torch.zeros(self.num_vehicles, len(self.hdv_indices))
        for i, hdv_idx in enumerate(self.hdv_indices):
            W_hdv[hdv_idx, i] = 1.0
        self.register_buffer('W_hdv', W_hdv)

        self.dt = 0.1
        # 状态转移矩阵A
        self.A = torch.zeros(2 * self.num_vehicles, 2 * self.num_vehicles, dtype=torch.float32)
        for i in range(0, 2 * self.num_vehicles):
            self.A[i, i] = 1
            #self.A[i+1, i+1] = 1
        for i in range(2, 2 * self.num_vehicles, 2):
            self.A[i, i+1] = -self.dt
            self.A[i, i-1] = self.dt

        # 控制输入矩阵B
        self.b = torch.zeros(2 * self.num_vehicles, self.num_vehicles, dtype=torch.float32)
        for vehicle_idx in range(1, self.num_vehicles):
            self.b[2*vehicle_idx + 1, vehicle_idx] = self.dt

        self.A = self.A.unsqueeze(0)
        self.b = self.b.unsqueeze(0)


    def forward(self, x):
        batch_size = 1
        
        # 创建参考状态
        x_stars = torch.tensor([[20.0, 15.0]] * self.num_vehicles, device=x.device)
        x_stars = x_stars.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 控制边界
        u_bounds = (torch.tensor(-5.0, device=x.device), 
                   torch.tensor(5.0, device=x.device))
        u_star = torch.zeros(1, device=x.device)
        
        # 获取CAV控制输出
        controllers_output = self.CombindedControllers(x, x_stars, u_star, u_bounds)

        # 使用选择矩阵组合加速度
        acceleration  = torch.matmul(self.W_cav, controllers_output.transpose(0,1)).transpose(0,1)

        # 计算下一个状态
        next_state = torch.matmul(self.A, x.reshape(2*self.num_vehicles)) + torch.matmul(self.b, acceleration.reshape(self.num_vehicles))
        
        # 计算Lyapunov值
        output_V = self.V_net(x, x_stars)
        next_V = self.V_net(next_state, x_stars)

        # Calculate barrier values
        output_B = self.B_net(x, x_stars)
        next_B = self.B_net(next_state, x_stars)

        return output_V, next_state, next_V, output_B, next_B

def combined_model(V_net, B_net, controllers, system_dynamics, output_file, state_dims, cav_indices): 
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
    
    # Create and export combined model
    combined_network = CombinedNetwork(controllers, V_net, B_net, cav_indices, state_dims, system_dynamics)
    
    torch.save(combined_network, output_file.replace(".onnx", ".pth"))
    # 创建包含所有车辆状态的dummy输入
    #dummy_input_x = torch.randn(1, len(state_dims), state_dims[0],requires_grad=True)  # [1, num_vehicles]
    dummy_input_x = torch.tensor([[[20,15],[21,14],[21,13],[21,13]]],requires_grad=True, dtype=torch.float32)  # [1, num_vehicles]
    
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
        output_names=['output_V','next_state','next_V', 'output_B', 'next_B'],
    )

    model = onnx.load(output_file)
    model_simp, check = simplify(model)
    onnx.save(model_simp, output_file)

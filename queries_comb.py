import os
from datetime import datetime
import numpy as np

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
import sys
sys.path.append("/barrett/scratch/udayanm/Marabou")
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils

from convertsinglenetwork import single_model
#from training_exp_mask import LyapunovNetworkV, TwoDimDocking

# from maraboupy import MarabouCore
class VerificationQuery:
    def __init__(self, network, num_agents=2):
        self.network = network
        self.num_agents = num_agents
        self.num_lyap = num_agents - 1
        
    def run_unroll(self, network):
        """单步验证的核心逻辑，处理多个Lyapunov函数
        """
        # 获取网络的输入和输出
        current_state = network.inputVars[0][0]  # 所有智能体的当前状态
        
        # 网络输出包含: [控制输出, V_1(x_t),...,V_{n-1}(x_t), x_{t+1}, V_1(x_{t+1}),...,V_{n-1}(x_{t+1})]
        network_output = network.outputVars[0][0]

        print(current_state)
        print(network_output)
        
        # 解析输出
        control_output = network_output[:2*self.num_agents]  
        v_current = network_output[2*self.num_agents:2*self.num_agents+self.num_lyap]  # 当前所有Lyapunov值
        next_state = network_output[2*self.num_agents+self.num_lyap:2*self.num_agents+self.num_lyap+4*self.num_agents]  # 下一状态
        v_next = network_output[-self.num_lyap:]  # 下一状态的所有Lyapunov值
        
        return current_state, control_output, next_state, v_current, v_next

    def check_descent(self, input_bounds, epsilon=0.01):
        """验证所有Lyapunov函数是否满足下降条件"""
        network = self.network
        
        # 获取所有变量
        current_state, control_output, next_state, v_current, v_next = self.run_unroll(network)
        
        # 设置输入范围约束
        for i, (lb, ub) in enumerate(input_bounds):
            network.setLowerBound(current_state[i], lb)
            network.setUpperBound(current_state[i], ub)
            
        # 对每个Lyapunov函数添加下降条件
        for i in range(self.num_lyap):
            descent_eq = MarabouUtils.Equation(MarabouCore.Equation.GT)
            descent_eq.addAddend(1, v_current[i])
            descent_eq.addAddend(-1, v_next[i])
            descent_eq.setScalar(epsilon)
            network.addEquation(descent_eq)
        
        # 求解验证问题
        vals, stats = network.solve()
        
        if len(vals) > 0:
            return [-1]  # 找到反例
        return [1] * 4  # 验证通过

def safe_descent_cond_check(PATH_TO_ONNX, x_star, prev_pos=4, limit_pos=5, vel_limit=0.5, num_agents=2):
    """主验证函数，处理多智能体系统
    Args:
        PATH_TO_ONNX: 组合模型的路径
        x_star: 目标状态
        prev_pos: 前一个位置边界
        limit_pos: 限制区域边界
        vel_limit: 速度限制
        num_agents: 智能体数量
    """
    # 加载组合后的ONNX模型
    network = Marabou.read_onnx(PATH_TO_ONNX)
    query = VerificationQuery(network, num_agents)
    
    # 定义空间划分（一维）
    spacing_space = np.linspace(-limit_pos, limit_pos, 5)
    velocity_space = np.linspace(-vel_limit, vel_limit, 5)
    
    # 验证结果存储
    vals = []
    val_ranges = []
    failed_vals = []
    
    # 对每个智能体的状态空间进行验证
    for i in range(4):
        for k in range(4):
            # 对每个智能体验证
            for agent in range(num_agents):
                state_bounds = []
                for _ in range(num_agents):
                    if _ == agent:
                        # 当前验证的智能体
                        state_bounds.extend([
                            [round(spacing_space[i], 2), round(spacing_space[i + 1], 2)],  # spacing
                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)]  # velocity
                        ])
                    else:
                        # 其他智能体
                        state_bounds.extend([
                            [-limit_pos, limit_pos],  # spacing
                            [-vel_limit, vel_limit]   # velocity
                        ])
                
                ans = query.check_descent(state_bounds)
                if len(ans) == 4:
                    vals.append(ans)
                    val_ranges.append(state_bounds)
                elif ans[0] == -1:
                    failed_vals.append(ans)
    
    return vals, val_ranges, len(failed_vals) == 0

if __name__ == "__main__":
    x_star = [20, 15]  # 目标状态：spacing=0, velocity=0
    vals, ranges, is_safe = safe_descent_cond_check("combined/combined_0.onnx", x_star, num_agents=3)
    print(f"Verification result: {'Safe' if is_safe else 'Unsafe'}")


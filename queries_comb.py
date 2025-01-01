import os
from datetime import datetime
import numpy as np

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
import sys
sys.path.append("/home/zhoujy53/Desktop/Marabou")
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils

from convertsinglenetwork import single_model
#from training_exp_mask import LyapunovNetworkV, TwoDimDocking

# from maraboupy import MarabouCore
class VerificationQuery:
    def __init__(self, network, system, num_agents=2):
        self.network = network
        self.num_agents = num_agents
        self.num_lyap = num_agents - 1
        self.system = system
        
    def run_unroll(self, network):
        """单步验证的核心逻辑，处理多个Lyapunov函数
        """
        # 获取网络的输入和输出
        current_state = network.inputVars[0][0]
        next_state = network.inputVars[1][0]
        
        # 网络输出包含: [控制输出, V_1(x_t),...,V_{n-1}(x_t), x_{t+1}, V_1(x_{t+1}),...,V_{n-1}(x_{t+1})]
        control_output = network.outputVars[0][0]
        v_current = network.outputVars[1][0]
        v_next = network.outputVars[2][0]

        #print("current_state", current_state)
        #print("next_state", next_state)
        #print("control_output", control_output)
        #print("v_current", v_current)
        #print("v_next", v_next)
        
        return current_state, next_state, control_output, v_current, v_next

    def check_descent(self, input_bounds, epsilon=0.01):
        """验证所有Lyapunov函数是否满足下降条件"""
        network = self.network

        useMILP = True
        options = Marabou.createOptions(verbosity=0, solveWithMILP=useMILP, snc=False)
        
        # 获取所有变量
        current_state, next_state, control_output, v_current, v_next = self.run_unroll(network)

        # 设置输入范围约束
        for agent in range(self.num_agents):

            # Set bounds for spacing
            network.setLowerBound(current_state[agent][0], input_bounds[agent * 2][0])
            network.setUpperBound(current_state[agent][0], input_bounds[agent * 2][1])
            network.setLowerBound(next_state[agent][0], input_bounds[agent * 2][0])
            network.setUpperBound(next_state[agent][0], input_bounds[agent * 2][1])
            
            # Set bounds for velocity
            network.setLowerBound(current_state[agent][1], input_bounds[agent * 2 + 1][0])
            network.setUpperBound(current_state[agent][1], input_bounds[agent * 2 + 1][1])
            network.setLowerBound(next_state[agent][1], input_bounds[agent * 2 + 1][0])
            network.setUpperBound(next_state[agent][1], input_bounds[agent * 2 + 1][1])
            
        # 对每个Lyapunov函数添加下降条件
        for i in range(self.num_lyap):
            aii = 0.6
            descent_eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
            descent_eq.addAddend(1, v_next[i])
            descent_eq.addAddend(-1, v_current[i])
            descent_eq.addAddend(aii, v_current[i])
            for j in self.system.connections[i+1]:
                descent_eq.addAddend(self.system.connections[i+1][j], v_current[j-1])
            descent_eq.setScalar(epsilon)
            network.addEquation(descent_eq)
        
        # 求解验证问题
        exitCode, vals, stats = network.solve(options=options, verbose=True)
        
        if exitCode == "sat":
            return [[vals[current_state[i][j]] for j in range(2)] for i in range(self.num_agents)]
        if exitCode == "unsat":
            return [1]
        else:
            return [-1]
    
        
def safe_descent_cond_check(PATH_TO_ONNX, system, limit_pos=40, vel_limit=30, num_agents=3):
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
    query = VerificationQuery(network, system, num_agents)
    
    # 定义空间划分（一维）
    spacing_space = np.linspace(0, limit_pos, 5)
    velocity_space = np.linspace(0, vel_limit, 5)

    dist_range = (-0.5,0.5)
    dist_space = np.linspace(dist_range[0], dist_range[1], 5)

    
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
                if len(ans) > 1:
                    vals.append(ans)
                    val_ranges.append(state_bounds)
                elif ans[0] == -1:
                    failed_vals.append(ans)
    
    return vals, val_ranges, len(failed_vals) == 5*5*num_agents

if __name__ == "__main__":
    x_star = [20, 15]  # 目标状态：spacing=0, velocity=0
    vals, ranges, failed = safe_descent_cond_check("/home/zhoujy53/Desktop/String_Stability_Verification/combined/combined_0.onnx", num_agents=3)
    print(f"Verification result: {'fail' if failed else 'succeed'}")


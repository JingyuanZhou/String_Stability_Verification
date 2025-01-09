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
    def __init__(self, PATH, system, num_agents=2):
        self.PATH = PATH
        self.num_agents = num_agents
        self.num_lyap = num_agents - 1
        self.system = system
        self.dt = 0.1
        
    def run_unroll(self, network):
        """单步验证的核心逻辑，处理多个Lyapunov函数的下降条件验证
        """
        # 获取网络的输入和输出
        #print("network.inputVars =", network.inputVars)
        #print("network.outputVars =", network.outputVars)

        current_state = network.inputVars[0][0]
        
        # 网络输出包含: [控制输出, V_1(x_t),...,V_{n-1}(x_t), x_{t+1}, V_1(x_{t+1}),...,V_{n-1}(x_{t+1})]
        #print(network.outputVars)
        v_current = network.outputVars[0][0]
        #print("v_current", v_current)
        next_state = network.outputVars[1]
        v_next = network.outputVars[2][0]

        #print("current_state", current_state)
        #print("acceleration", acceleration)
        #print("v_current", v_current)
        #print("next_state", next_state)
        #print("v_next", v_next)
        
        return current_state, next_state, v_current, v_next

        
    def check_descent(self, input_bounds, epsilon=0.01, useMILP=True):
        """
        验证给定输入范围下是否存在违反“李雅普诺夫下降条件”的反例。
        
        Args:
            input_bounds: 对所有车辆的 [spacing, velocity] 上下界 (list of lists)
                          结构类似:
                          [
                            [min_spacing_车0, max_spacing_车0],
                            [min_velocity_车0, max_velocity_车0],
                            [min_spacing_车1, max_spacing_车1],
                            [min_velocity_车1, max_velocity_车1],
                            ...
                          ]
            epsilon: 下降条件中的右侧常数项
            useMILP: 是否开启 MILP 求解器
        Returns:
            如果 exitCode=="sat"，返回找到的反例状态；
            如果 exitCode=="unsat"，返回 [1] 表示未找到反例；
            如果出现 timeout/error 等其他情况，返回 [-1]。
        """
        network = Marabou.read_onnx(self.PATH)
        options = Marabou.createOptions(
            verbosity=0,
            solveWithMILP=useMILP,
            snc=False
        )
        
        #print("input_bounds", input_bounds)

        current_state, next_state, v_current, v_next = self.run_unroll(network)
        #current_state, v_current = self.run_unroll(network)

        # ========== 1) 设置输入上下界，包含头车固定到平衡态的示例 ========== 
        # 假设 "头车(Agent=0)" 的 equilibrium spacing = 0, equilibrium velocity = 15
        # 你可根据实际需要修改
        eq_spacing = 20.0
        eq_velocity = 15.0
        
        # 对所有 agent 设置上下界
        for agent_id in range(self.num_agents):
            if agent_id == 0:
                # 头车固定到平衡态
                # current_state[0][0] -> spacing_头车
                # current_state[0][1] -> velocity_头车
                # next_state[0][0], next_state[0][1] 同理
                for var_idx in [
                    current_state[agent_id][0],
                    current_state[agent_id][1],
                    #next_state[agent_id][0],
                    #next_state[agent_id][1]
                ]:
                    network.setLowerBound(var_idx, eq_spacing if (var_idx % 2 == 0) else eq_velocity)
                    network.setUpperBound(var_idx, eq_spacing if (var_idx % 2 == 0) else eq_velocity)
                # control_output
            else:
                # 对后续车辆按照 input_bounds 给出的范围设置
                # agent_id 从 1 ~ (num_agents-1)
                # 在 input_bounds 中，车 i 对应 2*i, 2*i+1
                sp_lo, sp_hi = input_bounds[2*agent_id]
                vel_lo, vel_hi = input_bounds[2*agent_id+1]

                # current
                network.setLowerBound(current_state[agent_id][0], sp_lo)
                network.setUpperBound(current_state[agent_id][0], sp_hi)
                network.setLowerBound(current_state[agent_id][1], vel_lo)
                network.setUpperBound(current_state[agent_id][1], vel_hi)

        # ========== 2) 添加“李雅普诺夫函数下降”约束 ========== 
        # 假设 num_lyap = num_agents - 1，对应第 1 ~ (num_agents-1) 这几辆车
        # 下面只演示1个Lyapunov对每辆车的情况，如果你每辆车都输出了不同的 V_current[i], V_next[i],
        # 需要做更加细粒度的索引处理
        # 这里只做一个演示，示意如何写不等式

        for i in range(self.num_lyap):
            # Positive constraint: v_current[i] >= 0

            #network.setLowerBound(v_current[i], 0.0)
            #network.setLowerBound(v_next[i], 0.0)

            # Descent constraint
            
            aii = 0.6
            epsilon = -0.01
            vars = [v_next[i], v_current[i]]
            coeffs = [1.0, -1.0 + aii]  # Coefficients for v_next[i] and v_current[i]

            # Add coefficients for system.connections[i+1]
            for j in self.system.connections[i+1]:
                vars.append(v_current[j-1])  # Add v_current[j-1] variable
                coeffs.append(-self.system.connections[i+1][j])  # Corresponding coefficient

            # Add the inequality to the network
            network.addInequality(
                vars=vars,
                coeffs=coeffs,
                scalar=epsilon
            )
            


        # ========== 3) 调用 Marabou solver 验证 ========== 
        exitCode, vals, stats = network.solve(options=options, verbose=False)

        if exitCode == "sat":
            # 找到满足(违反下降条件)的反例 => 返回反例的状态 (仅示例：返回每辆车的 spacing, velocity)
            counterexample = []
            for agent_id in range(self.num_agents):
                spacing_val = vals[current_state[agent_id][0]]
                velocity_val = vals[current_state[agent_id][1]]
                counterexample.append([spacing_val, velocity_val])
            return counterexample  # 多维列表
        elif exitCode == "unsat":
            # 不可满足 => 不存在反例 => 安全
            return [1]
        elif exitCode == "timeout":
            # 超时，不确定
            return [-1]
        else:
            # 其他错误(可能 error)
            return [-1]
    

def safe_descent_cond_check(
    PATH_TO_ONNX,
    system,
    limit_pos=40,
    vel_limit=30,
    num_agents=3
):
    """
    主验证函数，尝试在一个离散网格上，对所有后车(agent=1..num_agents-1)的
    (spacing, velocity) 范围做遍历，并在每个网格上调用 check_descent()。
    
    - 假设头车(agent=0)在 check_descent() 里被固定到 equilibrium
    - 对后车范围做一个简单离散划分
    """
    # 1) 读入网络
    query = VerificationQuery(PATH_TO_ONNX, system, num_agents)

    # 2) 定义离散网格
    # 假设我们希望在 [0, limit_pos]、[0, vel_limit] 范围各划分 5 等份
    # => spacing_space: [0, 10, 20, 30, 40], velocity_space: [0, 7.5, 15, 22.5, 30]
    # => 4 个区间(因为有5个端点)
    split_num = 13
    spacing_space = np.linspace(5, 35, split_num)
    velocity_space = np.linspace(0, vel_limit, split_num)

    # 3) 存储验证结果
    vals_found = []       # 用来记录找到的反例
    val_ranges = []       # 记录对应的区间
    failed_vals = []      # 记录 timeout/error 等情况

    # 4) 两重循环: i in [0..3], k in [0..3] => 16个 (spacing, velocity) 区间
    for i in range(len(spacing_space) - 1):   # 0..3
        for k in range(len(velocity_space) - 1):  # 0..3
            for agent in range(1,num_agents):
                # 为 num_agents=3, 构造 input_bounds
                # agent=0 会在 check_descent() 里固定，所以这里只要给 agent=0 占位即可
                # agent=1,2 用实际区间
                # 结构: [ [sp_min_0, sp_max_0], [vel_min_0, vel_max_0],
                #         [sp_min_1, sp_max_1], [vel_min_1, vel_max_1],
                #         [sp_min_2, sp_max_2], [vel_min_2, vel_max_2] ]
                # 但头车(0)固定 => 可以给一个“fake”区间(后面不使用)
                if agent == 1:
                    state_bounds = [
                        [20, 20],   # spacing_头车
                        [15, 15],   # velocity_头车
                        # agent=1
                        [round(spacing_space[i], 2),   round(spacing_space[i+1], 2)],
                        [round(velocity_space[k], 2),  round(velocity_space[k+1], 2)],
                        # agent=2
                        [5, 35],
                        [0, vel_limit]
                    ]
                elif agent == 2:
                    state_bounds = [
                        [20, 20],   # spacing_头车
                        [15, 15],   # velocity_头车
                        # agent=1
                        [5, 35],
                        [0, vel_limit],
                        # agent=2
                        [round(spacing_space[i], 2),   round(spacing_space[i+1], 2)],
                        [round(velocity_space[k], 2),  round(velocity_space[k+1], 2)]
                    ]

                # 调用 check_descent 
                ans = query.check_descent(state_bounds)
                
                # 根据返回值分类
                if isinstance(ans, list) and len(ans) > 1:
                    # sat => ans 是反例
                    vals_found.append(ans)
                    val_ranges.append(state_bounds)
                elif ans[0] == -1:
                    # 其他错误 or 超时
                    failed_vals.append(ans)
                # 如果 ans = [1], 表示 "unsat" => 这一块区间无反例


    found_count = len(vals_found)     # 有反例的次数
    timeout_count = len(failed_vals)  # 超时或错误
    
    # 这里定义：只要存在反例 => fail； 或者存在超时 => 不确定
    # 你也可以根据自己需求修改
    if found_count > 0:
        verification_result = "fail (found at least one counterexample)"
    elif timeout_count > 0:
        verification_result = "inconclusive (some queries timed out or solver error)"
    else:
        verification_result = "succeed (no counterexamples found, no timeouts)"

    return vals_found, val_ranges, verification_result

if __name__ == "__main__":
    x_star = [20, 15]  # 目标状态：spacing=0, velocity=0
    vals, ranges, failed = safe_descent_cond_check("/home/zhoujy53/Desktop/String_Stability_Verification/combined/combined_0.onnx", num_agents=3)
    print(f"Verification result: {'fail' if failed else 'succeed'}")


import os
from datetime import datetime
import numpy as np
import torch

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
import sys
sys.path.append("/home/zhoujy53/Desktop/Marabou")
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils


#from training_exp_mask import LyapunovNetworkV, TwoDimDocking

# from maraboupy import MarabouCore
class VerificationQuery:
    def __init__(self, PATH, system, num_agents=2):
        self.PATH = PATH
        self.num_agents = num_agents
        self.num_lyap = num_agents - 1
        self.system = system
        self.dt = 0.1
        self.count_superious_ce = 0
        
    def run_unroll(self, network):
        """单步验证的核心逻辑，处理多个Lyapunov函数的下降条件验证
        """
        # 获取网络的输入和输出
        #print("network.inputVars =", network.inputVars)
        #print("network.outputVars =", network.outputVars)

        current_state = network.inputVars[0][0]
        
        # 网络输出包含: [控制输出, V_1(x_t),...,V_{n-1}(x_t), x_{t+1}, V_1(x_{t+1}),...,V_{n-1}(x_{t+1})]
        v_current = network.outputVars[0][0]
        next_state = network.outputVars[1]
        v_next = network.outputVars[2][0]
        b_current = network.outputVars[3][0]
        b_next = network.outputVars[4][0]

        #print("current_state", current_state)
        #print("v_current", v_current)
        #print("next_state", next_state)
        #print("v_next", v_next)
        
        return current_state, next_state, v_current, v_next, b_current, b_next

        
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
            solveWithMILP=useMILP, #
            snc=False,
            numWorkers=4
        )
        
        #print("input_bounds", input_bounds)

        current_state, next_state, v_current, v_next, b_current, b_next = self.run_unroll(network)
        #current_state, v_current = self.run_unroll(network)

        # ========== 1) 设置输入上下界，包含头车固定到平衡态的示例 ========== 
        # 假设 "头车(Agent=0)" 的 equilibrium spacing = 20, equilibrium velocity = 15
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

                network.setLowerBound(current_state[agent_id][0], eq_spacing)
                network.setUpperBound(current_state[agent_id][0], eq_spacing)
                network.setLowerBound(current_state[agent_id][1], eq_velocity)
                network.setUpperBound(current_state[agent_id][1], eq_velocity)
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

                #print("current_state[agent_id][0]", current_state[agent_id][0], "sp_lo", sp_lo, "sp_hi", sp_hi)
                #print("current_state[agent_id][1]", current_state[agent_id][1], "vel_lo", vel_lo, "vel_hi", vel_hi)
                

        # ========== 2) 添加“李雅普诺夫函数下降”约束 ========== 
        # 假设 num_lyap = num_agents - 1，对应第 1 ~ (num_agents-1) 这几辆车
        # 下面只演示1个Lyapunov对每辆车的情况，如果你每辆车都输出了不同的 V_current[i], V_next[i],
        # 需要做更加细粒度的索引处理
        # 这里只做一个演示，示意如何写不等式
        disjunction = []
        use_lyap = False
        if use_lyap:
            for i in range(self.num_lyap):
                # Positive constraint: v_current[i] >= 0

                ineq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
                ineq1.addAddend(1.0, v_current[i])
                ineq1.setScalar(0.000001) 

                ineq2 = MarabouUtils.Equation(MarabouCore.Equation.LE)
                ineq2.addAddend(1.0, v_next[i])
                ineq2.setScalar(0.000001) 

                # Descent constraint
                ineq3 = MarabouUtils.Equation(MarabouCore.Equation.GE)
                aii = self.system.connections[i+1][i+1]
                epsilon = -0.00001#0.001
                ineq3.addAddend(1.0, v_next[i])
                ineq3.addAddend(-1.0 + aii, v_current[i])
                for j in self.system.connections[i+1]:
                    if j >= 1:
                        ineq3.addAddend(-self.system.connections[i+1][j], v_current[j-1])

                ineq3.setScalar(epsilon) #epsilon

                disjunction.append([ineq1])
                #disjunction_lyap.append([ineq2])
                disjunction.append([ineq3])



        for i in range(self.num_lyap):
            epsilon_CBF = -0.000001
            ineq_CBF_positive = MarabouUtils.Equation(MarabouCore.Equation.GE)
            ineq_CBF_positive.addAddend(1.0, b_current[i])
            ineq_CBF_positive.setScalar(-epsilon_CBF)
            ineq_safety_con_positive = MarabouUtils.Equation(MarabouCore.Equation.GE)
            ineq_safety_con_positive.addAddend(1.0, current_state[i+1,0])
            ineq_safety_con_positive.addAddend(-0.5, current_state[i+1,1])
            ineq_safety_con_positive.setScalar(-epsilon_CBF)

            ineq_CBF_negative = MarabouUtils.Equation(MarabouCore.Equation.LE)
            ineq_CBF_negative.addAddend(1.0, b_current[i])
            ineq_CBF_negative.setScalar(epsilon_CBF) 
            ineq_safety_con_negative = MarabouUtils.Equation(MarabouCore.Equation.LE)
            ineq_safety_con_negative.addAddend(1.0, current_state[i+1,0])
            ineq_safety_con_negative.addAddend(-0.5, current_state[i+1,1])
            ineq_safety_con_negative.setScalar(epsilon_CBF)

            condition_1 = [ineq_CBF_positive, ineq_safety_con_negative]
            condition_2 = [ineq_CBF_negative, ineq_safety_con_positive]

            ineq_CBF_derivative = MarabouUtils.Equation(MarabouCore.Equation.GE)
            ineq_CBF_derivative.addAddend(-1.0, b_next[i])
            ineq_CBF_derivative.addAddend(1.0 + self.system.CBF_coupling_matrix[i][i], b_current[i])
            for j in self.system.CBF_coupling_matrix[i]:
                if j != i:
                    ineq_CBF_derivative.addAddend(self.system.CBF_coupling_matrix[i][j], b_current[j])

            ineq_CBF_derivative.setScalar(-epsilon_CBF)

            disjunction.append(condition_1)
            disjunction.append(condition_2)
            disjunction.append([ineq_CBF_derivative])

        network.addDisjunctionConstraint(disjunction)
        exitCode, vals, stats = network.solve(options=options, verbose=False)

        if exitCode == "sat":
            # 找到满足(违反下降条件)的反例 => 返回反例的状态 (仅示例：返回每辆车的 spacing, velocity)
            counterexample = []
            for agent_id in range(self.num_agents):
                spacing_val = vals[current_state[agent_id][0]]
                velocity_val = vals[current_state[agent_id][1]]
                counterexample.append([spacing_val, velocity_val])


            lya_current = [vals[v_current[i]] for i in range(self.num_lyap)]
            lya_next = [vals[v_next[i]] for i in range(self.num_lyap)]
            solved_next_state = [vals[next_state[i]] for i in range(self.num_agents*2)]

            ground_true = network.evaluateWithoutMarabou([np.array(counterexample)])
            # check counter example
            # decresing conditions
            '''
            ce = True
            expr_ls = []
            for i in range(self.num_lyap):
                aii = 0.05
                epsilon = 0.0
                vars_ = [lya_next[i], lya_current[i]]
                coeffs = [1.0, -1.0 + aii]

                for j in self.system.connections[i+1]:
                    if j >= 1:
                        vars_.append(lya_current[j-1])
                        coeffs.append(-self.system.connections[i+1][j])

                expr = sum(v * c for v, c in zip(vars_, coeffs))
                expr_ls.append(expr)
                if lya_current[0]<=0 or lya_current[1]<=0 or expr >= epsilon:
                    ce = False
            if ce:
                self.count_superious_ce += 1
            #print("ground_true", ground_true)

                print("counter_example", counterexample, "solved_next_state", solved_next_state,"lya_current", lya_current, "lya_next", lya_next, "expr_ls", expr_ls)
            '''
            #print("Find counter_example", counterexample)
            print("counter_example", counterexample, "solved_next_state", solved_next_state,"lya_current", lya_current, "lya_next", lya_next)
            return counterexample  # 多维列表
        elif exitCode == "unsat":
            # 不可满足 => 不存在反例 => 安全
            return [1]
        elif exitCode == "timeout":
            # 超时，不确定
            print("timeout")
            return [-1]
        else:
            print("error")
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
    split_num = [3,3,3]
    max_spacing = 25
    min_spacing = 15
    max_vel = 20
    min_vel = 10

    spacing_space = []
    velocity_space = []
    for i in range(num_agents-1):
        spacing_space.append(np.linspace(min_spacing, max_spacing, split_num[i]))
        velocity_space.append(np.linspace(min_vel, max_vel, split_num[i]))


    # 3) 存储验证结果
    vals_found = []       # 用来记录找到的反例
    val_ranges = []       # 记录对应的区间
    failed_vals = []      # 记录 timeout/error 等情况

    # 4) 两重循环: i in [0..3], k in [0..3] => 个 (spacing, velocity) 区间
    
    idx = 0
    for i1 in range(len(spacing_space[0]) - 1):  # 0..3
        for j1 in range(len(velocity_space[0]) - 1):
            for i2 in range(len(spacing_space[1]) - 1):
                for j2 in range(len(velocity_space[1]) - 1):
                    for i3 in range(len(spacing_space[2]) - 1):
                        for j3 in range(len(velocity_space[2]) - 1):
                                # 5) 定义每个区间的上下界
                                state_bounds = [
                                    [20, 20],   # spacing_头车
                                    [15, 15],   # velocity_头车
                                    [round(spacing_space[0][i1], 2), round(spacing_space[0][i1+1], 2)],
                                    [round(velocity_space[0][j1], 2), round(velocity_space[0][j1+1], 2)],
                                    [round(spacing_space[1][i2], 2), round(spacing_space[1][i2+1], 2)],
                                    [round(velocity_space[1][j2], 2), round(velocity_space[1][j2+1], 2)],
                                    [round(spacing_space[2][i3], 2), round(spacing_space[2][i3+1], 2)],
                                    [round(velocity_space[2][j3], 2), round(velocity_space[2][j3+1], 2)],
                                ]

                                # 调用 check_descent 
                                ans = query.check_descent(state_bounds)
                                print("idx", idx)
                                idx += 1
                                # 根据返回值分类
                                if isinstance(ans, list) and len(ans) > 1:
                                    # sat => ans 是反例
                                    vals_found.append(ans)
                                    val_ranges.append(state_bounds)
                                elif ans[0] == -1:
                                    # 其他错误 or 超时
                                    failed_vals.append(ans)
                                # 如果 ans = [1], 表示 "unsat" => 这一块区间无反例，安全    


    found_count = len(vals_found)     # 有反例的次数
    timeout_count = len(failed_vals)  # 超时或错误
    
    # 这里定义：只要存在反例 => fail； 或者存在超时 => 不确定
    # 你也可以根据自己需求修改
    if found_count > 0:
        verification_result = "fail (found at least one counterexample)"
        print("found_count", found_count)
        print("count_superious_ce", query.count_superious_ce)
    elif timeout_count > 0:
        verification_result = "inconclusive (some queries timed out or solver error)"
    else:
        verification_result = "succeed (no counterexamples found, no timeouts)"

    return vals_found, val_ranges, verification_result

if __name__ == "__main__":
    cur_comb_file = "combined/combined_0.onnx"
    network = Marabou.read_onnx(cur_comb_file)

    inputs = np.array([[20.0, 15.0], [5.0, 10.0], [10.2, 10.0], [10.2, 10.0], [10.2, 10.0]])
    options = Marabou.createOptions(
        verbosity=2,
        solveWithMILP=True,
        snc=False
    )
    outputsMarabou = network.evaluateWithMarabou([inputs], options)
    #network.saveQuery("query_2.txt")
    #outputWMarabou = network.evaluateWithoutMarabou([inputs])
    #pytorch_model = torch.load("combined/combined.pth")
    #pytorch_output = pytorch_model(torch.tensor(inputs, dtype=torch.float32))
    print("outputsMarabou", outputsMarabou)
    #print("evaluateWithoutMarabou", outputWMarabou)
    #print("pytorch_output", pytorch_output)
    
#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

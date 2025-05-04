import os
from datetime import datetime
import numpy as np
import torch
import itertools
# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
import sys
sys.path.append("/home/zhoujy53/Desktop/Marabou")
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils
import torch


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

        #print("current_state", current_state)
        #print("v_current", v_current)
        #print("next_state", next_state)
        #print("v_next", v_next)
        
        return current_state, next_state, v_current, v_next

        
    def check_descent(self, input_bounds, epsilon=0.0000001, useMILP=True):
        """
        验证给定输入范围下是否存在违反"李雅普诺夫下降条件"的反例。
        
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

        current_state, next_state, v_current, v_next = self.run_unroll(network)
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
                

        # ========== 2) 添加"李雅普诺夫函数下降"约束 ========== 
        # 假设 num_lyap = num_agents - 1，对应第 1 ~ (num_agents-1) 这几辆车
        # 下面只演示1个Lyapunov对每辆车的情况，如果你每辆车都输出了不同的 V_current[i], V_next[i],
        # 需要做更加细粒度的索引处理
        # 这里只做一个演示，示意如何写不等式
        disjunction = []
        for i in range(self.num_lyap):
            # Positive constraint: v_current[i] >= 0
            epsilon = 0.005
            ineq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
            ineq1.addAddend(1.0, v_current[i])
            ineq1.setScalar(-epsilon) 

            ineq2 = MarabouUtils.Equation(MarabouCore.Equation.LE)
            ineq2.addAddend(1.0, v_next[i])
            ineq2.setScalar(-epsilon) 

            # Descent constraint
            ineq3 = MarabouUtils.Equation(MarabouCore.Equation.GE)
            aii = self.system.connections[i+1][i+1]

            ineq3.addAddend(1.0, v_next[i])
            ineq3.addAddend(-1.0 + aii, v_current[i])
            for j in self.system.connections[i+1]:
                if j >= 1 and j != i+1:
                    ineq3.addAddend(-self.system.connections[i+1][j], v_current[j-1])

            ineq3.setScalar(epsilon) #epsilon

            disjunction.append([ineq1])
            #disjunction.append([ineq2])
            disjunction.append([ineq3])

        #network.saveQuery("query_1.txt")
        #print("disjunction", disjunction)
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
            return [-1]
        else:
            # 其他错误(可能 error)
            return [-1]
    

def centralized_verification(
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
    split_num = [4,4,3,2]
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
                            for i4 in range(len(spacing_space[3]) - 1):
                                for j4 in range(len(velocity_space[3]) - 1):
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
                                        [20, 20],   # spacing_last_veh
                                        [15, 15],   # velocity_last_veh
                                        #[round(spacing_space[3][i4], 2), round(spacing_space[3][i4+1], 2)],
                                        #[round(velocity_space[3][j4], 2), round(velocity_space[3][j4+1], 2)]
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

class DecentralizedVerificationQuery:
    def __init__(self, PATH, system, agent_id, num_inverters=3):
        """
        Initialize a decentralized verification query for microgrid
        
        Args:
            PATH: Path to the ONNX model
            system: Microgrid system model
            agent_id: The specific inverter this query is for (1 or 2 for followers)
            num_inverters: Number of inverters in the microgrid
        """
        self.PATH = PATH
        self.agent_id = agent_id  # The specific inverter this query is for
        self.num_inverters = num_inverters
        self.system = system
        self.dt = 0.01
        
    def run_unroll(self, network):
        """Decentralized verification logic for a single inverter"""
        current_state = network.inputVars[0][0]
        v_current = network.outputVars[0][0]
        next_state = network.outputVars[1]
        v_next = network.outputVars[2][0]
        
        return current_state, next_state, v_current, v_next

    def check_descent(self, input_bounds, epsilon=0.000001, useMILP=True):
        """
        Verify Lyapunov descent condition for a single inverter considering only its neighbors.
        
        Args:
            input_bounds: Bounds for the states of all inverters [delta, omega, xi]
            epsilon: Descent condition constant
            useMILP: Whether to use MILP solver
        """
        network = Marabou.read_onnx(self.PATH)
        options = Marabou.createOptions(
            verbosity=0,
            solveWithMILP=True,
            snc=False,
            numWorkers=4,
            timeoutInSeconds=30
        )

        current_state, next_state, v_current, v_next = self.run_unroll(network)

        # Get neighbors of the current inverter
        neighbors = [j for j in self.system.connections[self.agent_id].keys()]
        
        # Set bounds for all inverters in the microgrid
        for inv_id in range(self.num_inverters):
            # For each inverter, set bounds on delta, omega, xi
            for i in range(3):
                lo, hi = input_bounds[inv_id*3 + i]
                network.setLowerBound(current_state[inv_id*3 + i], lo)
                network.setUpperBound(current_state[inv_id*3 + i], hi)

        # Add Lyapunov descent constraint only for this inverter
        disjunction = []
        
        # Positive constraint: V(x) > epsilon
        ineq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        ineq1.addAddend(1.0, v_current[self.agent_id])
        ineq1.setScalar(-epsilon)
        
        # Descent constraint considering only neighbors
        ineq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        
        # Get self-decay coefficient
        aii = self.system.connections[self.agent_id][self.agent_id]
        
        # Add terms for Lyapunov function decrease condition
        ineq2.addAddend(1.0, v_next[self.agent_id])
        ineq2.addAddend(-1.0 + aii, v_current[self.agent_id])

        # Add influence from neighboring inverters
        for j in self.system.connections[self.agent_id]:
            if j >= 1 and j != self.agent_id:
                # For each neighbor (except leader and self), add its influence
                ineq2.addAddend(-self.system.connections[self.agent_id][j], v_current[j])
                
        ineq2.setScalar(epsilon)

        # Add constraints to the disjunction
        disjunction.append([ineq1])
        disjunction.append([ineq2])
        
        network.addDisjunctionConstraint(disjunction)
        exitCode, vals, stats = network.solve(options=options, verbose=False)

        if exitCode == "sat":
            # Found counterexample - return states of all inverters
            counterexample = []
            for inv_id in range(self.num_inverters):
                delta = vals[current_state[inv_id*3 + 0]]
                omega = vals[current_state[inv_id*3 + 1]]
                xi = vals[current_state[inv_id*3 + 2]]
                counterexample.append([delta, omega, xi])
            
            # Get Lyapunov values for analysis
            lya_current = vals[v_current[self.agent_id]]
            lya_next = vals[v_next[self.agent_id]]
            
            # Verify constraint violation for debugging
            expr_ls = []
            aii = self.system.connections[self.agent_id][self.agent_id]
            vars_ = [lya_next, lya_current]
            coeffs = [1.0, -1.0 + aii]
            
            # Add terms for neighbors
            for j in self.system.connections[self.agent_id]:
                if j != self.agent_id:
                    vars_.append(vals[v_current[j]])
                    coeffs.append(-self.system.connections[self.agent_id][j])
                    
            # Calculate the expression value
            expr = sum(v * c for v, c in zip(vars_, coeffs))
            expr_ls.append(expr)
            next_state_values = [vals[next_state[0][i]] for i in range(self.num_inverters*3)]
            
            print(f"state: {counterexample}, next_state: {next_state_values}, lya_current: {lya_current}, lya_next: {lya_next}, expr: {expr_ls}")
            
            omega_star = 2 * torch.pi * 50  # Nominal frequency (50 Hz)
            counterexample[0][1] += omega_star
            counterexample[1][1] += omega_star
            counterexample[2][1] += omega_star

            return counterexample
        elif exitCode == "unsat":
            # No counterexample found - property holds
            return [1]
        else:
            # Verification failed
            return [-1]

def decentralized_verification(
    PATH_TO_ONNX,
    system,
    num_inverters=3,
    ret_ranges=None
):
    """
    Verify stability of the entire microgrid by checking each inverter's Lyapunov function.
    
    Args:
        onnx_path: Path to the ONNX model
        system: Microgrid system model
        delta_range: Range of delta values for each inverter
        omega_error_range: Range of omega error values for each inverter
        xi_range: Range of xi values for each inverter
        omega_star: Nominal frequency
    """
    vals_found = []
    val_ranges = []
    failed_vals = []
    results = []
    omega_star = 2 * np.pi * 50  # Nominal frequency (50 Hz)
    
    # Verify each follower inverter separately
    for inv_id in range(num_inverters):
        print(f"\n=== Verifying Follower Inverter {inv_id} ===")
        query = DecentralizedVerificationQuery(PATH_TO_ONNX, system, inv_id, num_inverters)
        
        # Get neighbors for this inverter
        neighbors = [j for j in system.connections[inv_id].keys()]
        print(f"Inverter {inv_id} has neighbors: {neighbors}")
        
        # Create grid divisions with different resolutions for each parameter
        split_nums = [4,4,4]  # Different resolutions to try
        
        # Define angle and frequency error ranges for verification
        delta_range = []
        omega_error_range = []
        xi_range = []
        
        for k_ in range(len(split_nums)):
            delta_range.append(np.linspace(0, np.pi/4, split_nums[k_]))  # Phase angle error range (rad)
            omega_error_range.append(np.linspace(-50, 50.0, split_nums[k_]))  # Frequency error range (rad/s)
            xi_range.append(np.array([0,0]))    # Controller state error range np.linspace(0, 5, split_nums[k])

        # Helper function to generate all combinations of neighbor bounds
        def generate_neighbor_bounds(index):
            # Initialize bounds with equilibrium values for all inverters
            # For each inverter: [delta, omega, xi]
            base_bounds = []
            
            # Initialize all dimensions for all inverters with equilibrium values
            for i in range(num_inverters):
                    # Delta bounds (phase angle)
                    base_bounds.append([0.0, 0.0])  # Equilibrium phase angle
                    
                    # Omega bounds (frequency)
                    base_bounds.append([0.0, 0.0])  # Equilibrium frequency
                    
                    # Xi bounds (secondary controller state)
                    base_bounds.append([0.0, 0.0])  # Equilibrium controller state
            
            # Generate indices and ranges for each neighbor's states
            neighbor_indices = []
            neighbor_ranges = []

            neighbor_indices.extend([
                (inv_id, 'delta'), 
                (inv_id, 'omega'), 
                (inv_id, 'xi')
            ])
            neighbor_ranges.extend([
                range(len(delta_range[index])-1),
                range(len(omega_error_range[index])-1),
                range(len(xi_range[index])-1)
            ])
            # First add the inverter being verified (we'll vary all its states)
            if inv_id < num_inverters-1:
                neighbor_indices.extend([
                    (inv_id+1, 'delta'), 
                    (inv_id+1, 'omega'), 
                    (inv_id+1, 'xi')
                ])
                neighbor_ranges.extend([
                    range(len(delta_range[index])-1),
                    range(len(omega_error_range[index])-1),
                    range(len(xi_range[index])-1)
                ])
            elif inv_id > 0:
                neighbor_indices.extend([
                    (inv_id-1, 'delta'), 
                    (inv_id-1, 'omega'), 
                    (inv_id-1, 'xi')
                ])
                neighbor_ranges.extend([
                    range(len(delta_range[index])-1),
                    range(len(omega_error_range[index])-1),
                    range(len(xi_range[index])-1)
                ])

            # Generate all combinations of grid points for the inverter being verified and its neighbors
            for idx_combination in itertools.product(*neighbor_ranges):
                current_bounds = base_bounds.copy()
                
                # Apply each index to the corresponding neighbor's states
                for (neighbor_idx, state_type), grid_idx in zip(neighbor_indices, idx_combination):
                    if state_type == 'delta':
                        # Update delta bounds
                        delta_idx = neighbor_idx * 3  # Each inverter has 3 states
                        delta_lo = delta_range[index][grid_idx]
                        delta_hi = delta_range[index][grid_idx + 1]
                        current_bounds[delta_idx] = [delta_lo, delta_hi]
                    elif state_type == 'omega':
                        # Update omega bounds (add error to nominal frequency)
                        omega_idx = neighbor_idx * 3 + 1
                        omega_error_lo = omega_error_range[index][grid_idx]
                        omega_error_hi = omega_error_range[index][grid_idx + 1]
                        current_bounds[omega_idx] = [
                            omega_error_lo,
                            omega_error_hi
                        ]
                    elif state_type == 'xi':
                        # Update xi bounds
                        xi_idx = neighbor_idx * 3 + 2
                        xi_lo = xi_range[index][grid_idx]
                        xi_hi = xi_range[index][grid_idx + 1]
                        current_bounds[xi_idx] = [xi_lo, xi_hi]

                yield current_bounds

        # Verify each combination of neighbor states
        idx = 0
        inv_vals_found = []
        inv_failed_vals = []
        
        for state_bounds in generate_neighbor_bounds(inv_id):
            print(f'Verifying Inverter {inv_id}, combination {idx}')
            # Run verification query
            ans = query.check_descent(state_bounds)
            idx += 1
            
            if isinstance(ans, list) and len(ans) > 1:
                # Found counterexample
                inv_vals_found.append(ans)
                vals_found.append(ans)
                val_ranges.append(state_bounds)
            elif ans[0] == -1:
                # Verification failed
                inv_failed_vals.append(ans)
                failed_vals.append(ans)

        # Determine verification result for this inverter
        if len(inv_vals_found) > 0:
            result = f"Inverter {inv_id}: Failed (found {len(inv_vals_found)} counterexamples)"
        elif len(inv_failed_vals) > 0:
            result = f"Inverter {inv_id}: Inconclusive ({len(inv_failed_vals)} queries failed)"
        else:
            result = f"Inverter {inv_id}: Succeeded (stability verified)"

        results.append(result)
        print(result)
    
    # Determine overall verification result
    if len(vals_found) > 0:
        overall_result = f"Microgrid: Failed (found {len(vals_found)} counterexamples total)"
    elif len(failed_vals) > 0:
        overall_result = f"Microgrid: Inconclusive ({len(failed_vals)} queries failed total)"
    else:
        overall_result = f"Microgrid: Succeeded (stability verified for all inverters)"
    
    results.append(overall_result)
    print(f"\n{overall_result}")
    
    return vals_found, val_ranges, results

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

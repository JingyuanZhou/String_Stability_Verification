import os
from datetime import datetime
import numpy as np
import torch
import itertools
# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
import sys
sys.path.append("/home/jy/Marabou")
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
    def __init__(self, PATH, system, agent_id, num_uavs=3):
        """
        Initialize a decentralized verification query for UAV formation
        
        Args:
            PATH: Path to the ONNX model
            system: UAV system model
            agent_id: The specific UAV this query is for (1 or 2 for followers)
            num_uavs: Number of UAVs in the formation (3 for leader + 2 followers)
        """
        self.PATH = PATH
        self.agent_id = agent_id  # The specific agent this query is for
        self.num_uavs = num_uavs
        self.system = system
        self.dt = 0.1
        self.dim = 3  # 3D space
        
    def run_unroll(self, network):
        """Decentralized verification logic for a single agent
        """
        current_state = network.inputVars[0][0]
        v_current = network.outputVars[0][0]
        next_state = network.outputVars[1]
        v_next = network.outputVars[2][0]
        
        return current_state, next_state, v_current, v_next

    def check_descent(self, input_bounds, epsilon=0.000001, useMILP=True):
        """
        Verify Lyapunov descent condition for a single UAV considering only its neighbors.
        
        Args:
            input_bounds: Bounds only for the states of the UAV and its neighbors
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

        # Get neighbors of the current UAV
        neighbors = [j for j in self.system.connections[self.agent_id].keys()]
        
        
        # Set bounds for all UAVs in the formation
        for uav_id in range(self.num_uavs):
            # For neighbors, use the provided bounds
            for i in range(6):
                lo, hi = input_bounds[uav_id*6 + i]
                network.setLowerBound(current_state[uav_id][i], lo)
                network.setUpperBound(current_state[uav_id][i], hi)

        # Add Lyapunov descent constraint only for this UAV
        disjunction = []
        
        # Positive constraint: V(x) > epsilon
        ineq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        ineq1.addAddend(1.0, v_current[self.agent_id-1])
        ineq1.setScalar(-epsilon)
        
        # Descent constraint considering only neighbors
        ineq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        
        # Get self-decay coefficient
        aii = self.system.connections[self.agent_id][self.agent_id]
        
        # Add terms for Lyapunov function decrease condition
        ineq2.addAddend(1.0, v_next[self.agent_id-1])
        ineq2.addAddend(-1.0 + aii, v_current[self.agent_id-1])

        # Add influence from neighboring UAVs
        for j in self.system.connections[self.agent_id]:
            if j >= 1 and j != self.agent_id:
                # For each neighbor (except leader and self), add its influence
                ineq2.addAddend(-self.system.connections[self.agent_id][j], v_current[j-1])
                
        ineq2.setScalar(epsilon)

        # Add constraints to the disjunction
        disjunction.append([ineq1])
        disjunction.append([ineq2])
        
        network.addDisjunctionConstraint(disjunction)
        exitCode, vals, stats = network.solve(options=options, verbose=False)

        if exitCode == "sat":
            # Found counterexample - return states of all UAVs
            counterexample = []
            for uav_id in range(self.num_uavs):
                pos_x = vals[current_state[uav_id][0]]
                pos_y = vals[current_state[uav_id][1]]
                pos_z = vals[current_state[uav_id][2]]
                vel_x = vals[current_state[uav_id][3]]
                vel_y = vals[current_state[uav_id][4]]
                vel_z = vals[current_state[uav_id][5]]
                counterexample.append([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z])
            
            # Get Lyapunov values for analysis
            lya_current = vals[v_current[self.agent_id-1]]
            lya_next = vals[v_next[self.agent_id-1]]
            # solved_next_state = [vals[next_state[self.agent_id-1][i]] for i in range(self.num_uavs*6)]
            
            # Verify constraint violation for debugging
            expr_ls = []
            aii = self.system.connections[self.agent_id][self.agent_id]
            vars_ = [lya_next, lya_current]
            coeffs = [1.0, -1.0 + aii]
            
            # Add terms for neighbors
            for j in self.system.connections[self.agent_id]:
                if j >= 1 and j != self.agent_id:
                    vars_.append(vals[v_current[j-1]])
                    coeffs.append(-self.system.connections[self.agent_id][j])
                    
            # Calculate the expression value
            expr = sum(v * c for v, c in zip(vars_, coeffs))
            expr_ls.append(expr)
            
            print(f"state: {counterexample}, lya_current: {lya_current}, lya_next: {lya_next}, expr: {expr_ls}")
            
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
    num_uavs=3,
    ret_ranges=None
):
    """
    Decentralized verification for a 3-UAV formation (1 leader + 2 followers).
    Verifies each follower UAV separately by checking its Lyapunov function
    considering only its connections to neighbors.
    
    Args:
        PATH_TO_ONNX: Path to the ONNX model
        system: UAV system model with connection information
        limit_pos: Position limit for verification region
        vel_limit: Velocity limit for verification region
        num_uavs: Number of UAVs in the formation (3 for leader + 2 followers)
        ret_ranges: Optional parameter for returning specific ranges
        
    Returns:
        vals_found: List of counterexamples found
        val_ranges: Corresponding state bounds for counterexamples
        results: Verification results for each UAV
    """
    vals_found = []
    val_ranges = []
    failed_vals = []
    results = []
    delta_ref = 10
    
    # Verify each follower UAV separately
    for agent_id in range(1, 2): #num_uavs
        print(f"\n=== Verifying Follower UAV {agent_id} ===")
        query = DecentralizedVerificationQuery(PATH_TO_ONNX, system, agent_id, num_uavs)
        
        # Get neighbors for this UAV
        neighbors = [j for j in system.connections[agent_id].keys()]
        print(f"UAV {agent_id} has neighbors: {neighbors}")
        
        # Create grid divisions with different resolutions for each UAV
        split_num = [10, 6]
        
        # Define spacing and velocity ranges for verification 
        # Focus only on x-direction errors
        x_spacing_error_range = []
        x_velocity_error_range = []
        for k in split_num:
            x_spacing_error_range.append(np.linspace(-3, 3, k) + delta_ref)  # x-position range
            x_velocity_error_range.append(np.linspace(-3, 3, k))  # x-velocity range

        # Helper function to generate all combinations of neighbor bounds
        def generate_neighbor_bounds(index):
            # Initialize bounds with equilibrium values for all UAVs
            # For each UAV: [x, y, z, vx, vy, vz]
            base_bounds = []
            
            # Initialize all dimensions for all UAVs with equilibrium values
            for uav_id in range(num_uavs):
                # Spacing bounds (x, y, z)
                for i in range(3):
                    if i == 0:  # x position
                        if uav_id == 0:
                            base_bounds.append([0.0, 0.0])  # Default x-error position (equilibrium spacing)
                        else:
                            base_bounds.append([delta_ref, delta_ref])    # Keep error position at 0
                    else:  # y, z positions
                        base_bounds.append([0.0, 0.0])    # Keep error position at 0
                
                # Velocity bounds (vx, vy, vz)
                for i in range(3):
                    if i == 0:  # x velocity
                        base_bounds.append([0.0, 0.0])  # Default x-error velocity (equilibrium)
                    else:  # vy, vz
                        base_bounds.append([0.0, 0.0])    # Keep error velocity at 0
            
            # Generate indices and ranges for each neighbor's states (x dimension only)
            neighbor_indices = []
            neighbor_ranges = []
            
            
            # Then add other relevant neighbors (excluding the leader)
            for neighbor in neighbors:
                if neighbor != 0:  # Skip leader and the UAV being verified (already added)
                    # For each neighbor, we only vary x-position and x-velocity
                    neighbor_indices.extend([(neighbor, 'x'), (neighbor, 'vx')])
                    neighbor_ranges.extend([
                        range(len(x_spacing_error_range[index])-1), 
                        range(len(x_velocity_error_range[index])-1)
                    ])
            
            # Generate all combinations of grid points for UAV being verified and its neighbors
            for idx_combination in itertools.product(*neighbor_ranges):
                current_bounds = base_bounds.copy()
                
                # Apply each index to the corresponding neighbor's x states
                for (neighbor_idx, state_type), grid_idx in zip(neighbor_indices, idx_combination):
                    if state_type == 'x':
                        # Update x-position bounds (first coordinate of each UAV)
                        x_pos_idx = neighbor_idx * 6  # Each UAV has 6 states
                        current_bounds[x_pos_idx] = [
                            x_spacing_error_range[index][grid_idx],
                            x_spacing_error_range[index][grid_idx + 1],
                        ]
                    elif state_type == 'vx':
                        # Update x-velocity bounds (fourth coordinate of each UAV)
                        vx_idx = neighbor_idx * 6 + 3  # Position index + 3 = velocity index
                        current_bounds[vx_idx] = [
                            x_velocity_error_range[index][grid_idx],
                            x_velocity_error_range[index][grid_idx + 1]
                        ]
                
                yield current_bounds

        # Verify each combination of neighbor states
        idx = 0
        agent_vals_found = []
        agent_failed_vals = []
        
        for error_state_bounds in generate_neighbor_bounds(agent_id-1):
            # transform error_state_bounds to state_bounds
            state_bounds = []
            for i in range(len(error_state_bounds)):
                # Get UAV index and dimension type
                uav_idx = i // 6  # Each UAV has 6 states [x,y,z,vx,vy,vz]
                dim_type = i % 6  # 0,1,2 = position, 3,4,5 = velocity
                
                if uav_idx == 0:
                    # Leader UAV (fixed reference)
                    if dim_type == 0:  # x position
                        state_bounds.append([0.0, 0.0])  # Leader at origin
                    elif dim_type == 1 or dim_type == 2:  # y, z position
                        state_bounds.append([0.0, 0.0])  # No y, z displacement
                    elif dim_type == 3:  # x velocity
                        state_bounds.append([5.0, 5.0])  # Fixed cruise velocity
                    else:  # vy, vz
                        state_bounds.append([0.0, 0.0])  # No y, z velocity
                else:
                    # Follower UAVs
                    if dim_type == 0:  # x position
                        # Calculate absolute position from spacing error
                        # Position = Leader position - (desired spacing + error)
                        if uav_idx>agent_id:
                            pos_lo = -delta_ref*(uav_idx)
                            pos_hi = -delta_ref*(uav_idx)
                        else:
                            error_lo, error_hi = error_state_bounds[i]
                            pos_lo = state_bounds[i-6][0] - error_hi  # Smaller error means larger negative position
                            pos_hi = state_bounds[i-6][1] - error_lo  # Larger error means smaller negative position
                        state_bounds.append([pos_lo, pos_hi])
                    elif dim_type == 1 or dim_type == 2:  # y, z position
                        state_bounds.append([0.0, 0.0])  # No y, z displacement
                    elif dim_type == 3:  # x velocity
                        # Calculate absolute velocity from velocity error
                        # Velocity = Leader velocity + error
                        if uav_idx>agent_id:
                            vel_lo = 5
                            vel_hi = 5
                        else:
                            error_lo, error_hi = error_state_bounds[i]
                            vel_lo = state_bounds[i-6][0] + error_lo
                            vel_hi = state_bounds[i-6][1] + error_hi
                        state_bounds.append([vel_lo, vel_hi])
                    else:  # vy, vz
                        state_bounds.append([0.0, 0.0])  # No y, z velocity
            
            print(f'Verifying UAV {agent_id}, combination {idx}')
            # Show bounds for this UAV and its neighbors
            
            # Run verification query
            ans = query.check_descent(state_bounds)
            idx += 1
            
            if isinstance(ans, list) and len(ans) > 1:
                # Found counterexample
                agent_vals_found.append(ans)
                vals_found.append(ans)
                val_ranges.append(state_bounds)
            elif ans[0] == -1:
                # Verification failed
                agent_failed_vals.append(ans)
                failed_vals.append(ans)

        # Determine verification result for this UAV
        if len(agent_vals_found) > 0:
            result = f"UAV {agent_id}: Failed (found {len(agent_vals_found)} counterexamples)"
        elif len(agent_failed_vals) > 0:
            result = f"UAV {agent_id}: Inconclusive ({len(agent_failed_vals)} queries failed)"
        else:
            result = f"UAV {agent_id}: Succeeded (string stability verified)"

        results.append(result)
        print(result)
    
    # Determine overall verification result
    if len(vals_found) > 0:
        overall_result = f"UAV Formation: Failed (found {len(vals_found)} counterexamples total)"
    elif len(failed_vals) > 0:
        overall_result = f"UAV Formation: Inconclusive ({len(failed_vals)} queries failed total)"
    else:
        overall_result = f"UAV Formation: Succeeded (string stability verified for all followers)"
    
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

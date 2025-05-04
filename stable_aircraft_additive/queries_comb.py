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

class DecentralizedVerificationQuery:
    def __init__(self, PATH, system, agent_id, num_uavs=3, shared_Lyapunov=True):
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
        self.shared_Lyapunov = shared_Lyapunov
    def run_unroll(self, network):
        """Decentralized verification logic for a single agent
        """
        current_state = network.inputVars[0][0]
        v_current = network.outputVars[0][0]
        next_state = network.outputVars[1]
        v_next = network.outputVars[2][0]
        
        return current_state, next_state, v_current, v_next

    def check_descent(self, input_bounds, epsilon=0.001, useMILP=True):
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
    ret_ranges=None,
    shared_Lyapunov=True,
    train_additive=False,
    load_additive_path=None,
    intitial_platoon_num=3,
    additive_platoon_num=3
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
    if train_additive:
        agent_range = [i for i in range(intitial_platoon_num, num_uavs)]
    else:
        agent_range = [i for i in range(1, num_uavs)]
    for agent_id in agent_range:
        print(f"\n=== Verifying Follower UAV {agent_id} ===")
        query = DecentralizedVerificationQuery(PATH_TO_ONNX, system, agent_id, num_uavs)
        
        # Get neighbors for this UAV
        neighbors = [j for j in system.connections[agent_id].keys()]
        print(f"UAV {agent_id} has neighbors: {neighbors}")
        
        split_num = [4 for i in range(num_uavs)]
        
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

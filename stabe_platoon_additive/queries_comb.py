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
    def __init__(self, PATH, system, agent_id, num_agents=2):
        self.PATH = PATH
        self.agent_id = agent_id  # The specific agent this query is for
        self.num_agents = num_agents
        self.system = system
        self.dt = 0.1
        
    def run_unroll(self, network):
        """Decentralized verification logic for a single agent
        """

        current_state = network.inputVars[0][0]
        v_current = network.outputVars[0][0]
        next_state = network.outputVars[1]
        v_next = network.outputVars[2][0]
        
        return current_state, next_state, v_current, v_next

    def check_descent(self, input_bounds, epsilon=0.0000001, useMILP=True):
        """
        Verify Lyapunov descent condition for a single agent considering only its neighbors.
        
        Args:
            input_bounds: Bounds only for the states of the agent and its neighbors
            epsilon: Descent condition constant
            useMILP: Whether to use MILP solver
        """
        network = Marabou.read_onnx(self.PATH)
        options = Marabou.createOptions(
            verbosity=0,
            solveWithMILP=useMILP,
            snc=False,
            numWorkers=4
        )

        current_state, next_state, v_current, v_next = self.run_unroll(network)

        # Get neighbors of the current agent
        neighbors = [j for j in self.system.connections[self.agent_id].keys()]
        
        network.setLowerBound(current_state[0][0], 20.0)  # eq_spacing
        network.setUpperBound(current_state[0][0], 20.0)
        network.setLowerBound(current_state[0][1], 15.0)  # eq_velocity
        network.setUpperBound(current_state[0][1], 15.0)
        # Set bounds only for the agent and its neighbors
        for agent_id in range(1, self.num_agents):
            if agent_id in neighbors:
                sp_lo, sp_hi = input_bounds[2*agent_id]
                vel_lo, vel_hi = input_bounds[2*agent_id+1]
                network.setLowerBound(current_state[agent_id][0], sp_lo)
                network.setUpperBound(current_state[agent_id][0], sp_hi)
                network.setLowerBound(current_state[agent_id][1], vel_lo)
                network.setUpperBound(current_state[agent_id][1], vel_hi)
            else:
                network.setLowerBound(current_state[agent_id][0], 20.0)  # eq_spacing
                network.setUpperBound(current_state[agent_id][0], 20.0)
                network.setLowerBound(current_state[agent_id][1], 15.0)  # eq_velocity
                network.setUpperBound(current_state[agent_id][1], 15.0)


        # Add Lyapunov descent constraint only for this agent
        disjunction = []
        
        # Positive constraint
        epsilon = 0.005
        ineq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        ineq1.addAddend(1.0, v_current[self.agent_id-1])
        ineq1.setScalar(-epsilon)
        
        # Descent constraint considering only neighbors
        ineq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        aii = self.system.connections[self.agent_id][self.agent_id]
        ineq2.addAddend(1.0, v_next[self.agent_id-1])
        ineq2.addAddend(-1.0 + aii, v_current[self.agent_id-1])

        # Only add terms for neighbors
        for j in self.system.connections[self.agent_id]:
            if j >= 1 and j != self.agent_id:
                ineq2.addAddend(-self.system.connections[self.agent_id][j], v_current[j-1])
                
        ineq2.setScalar(epsilon) #0.0000000001

        disjunction.append([ineq1])
        disjunction.append([ineq2])
        
        network.addDisjunctionConstraint(disjunction)
        exitCode, vals, stats = network.solve(options=options, verbose=False)

        if exitCode == "sat":
            # Found counterexample - return states of all agents
            counterexample = []
            for agent_id in range(self.num_agents):
                spacing_val = vals[current_state[agent_id][0]]
                velocity_val = vals[current_state[agent_id][1]]
                counterexample.append([spacing_val, velocity_val])
            
            lya_current = vals[v_current[self.agent_id-1]]
            lya_next = vals[v_next[self.agent_id-1]]
            solved_next_state = [vals[next_state[i]] for i in range(self.num_agents*2)]
            
            # check ineq 2
            expr_ls = []
            aii = self.system.connections[self.agent_id][self.agent_id]
            vars_ = [lya_next, lya_current]
            coeffs = [1.0, -1.0 + aii]
            for j in self.system.connections[self.agent_id]:
                if j >= 1 and j != self.agent_id:
                    vars_.append(vals[v_current[j-1]])
                    coeffs.append(-self.system.connections[self.agent_id][j])
            expr = sum(v * c for v, c in zip(vars_, coeffs))
            expr_ls.append(expr)
            #print("current_state", counterexample, "next_state", solved_next_state, "v_current", lya_current, "v_next", lya_next)
            print("lya_current", lya_current, "lya_next", lya_next, "expr_ls", expr_ls)
            return counterexample
        elif exitCode == "unsat":
            return [1]
        else:
            return [-1]

def decentralized_verification(
    PATH_TO_ONNX,
    system,
    limit_pos=40,
    vel_limit=30,
    num_agents=3,
    ret_ranges = None,
    shared_Lyapunov = False,
    train_additive = False,
    intitial_platoon_num = 5,
    additive_platoon_num = 5
):

    vals_found = []
    val_ranges = []
    failed_vals = []
    results = []
    # Verify each agent separately (except leader)
    for agent_id in range(1, num_agents):
        query = DecentralizedVerificationQuery(PATH_TO_ONNX, system, agent_id, num_agents)
        
        # Get neighbors for this agent
        neighbors = [j for j in system.connections[agent_id].keys()]
        
        # Create grid divisions
        # [3 5 5 1]
        if train_additive:
            split_num = [1 for _ in range(num_agents-1)]
            for k in range(intitial_platoon_num, intitial_platoon_num + additive_platoon_num-1):
                split_num[k] = 3
        else:
            split_num = [3 for _ in range(num_agents-1)]
        
        spacing_ranges = []
        velocity_ranges = []
        for k in split_num:
            spacing_ranges.append(np.linspace(15, 25, k))
            velocity_ranges.append(np.linspace(10, 20, k))
        
        # Create nested loops dynamically based on number of neighbors
        def generate_neighbor_bounds(index):
            # Initialize bounds with equilibrium values
            base_bounds = [[20.0, 20.0] if i % 2 == 0 else [15.0, 15.0] 
                         for i in range(num_agents * 2)]
            
            # Generate all combinations of grid points for neighbors
            neighbor_indices = []
            neighbor_ranges = []
            
            for neighbor in neighbors:
                if neighbor != 0:  # Skip leader as it's fixed
                    # For each non-leader neighbor, we need spacing and velocity indices
                    neighbor_indices.extend([(neighbor, 's'), (neighbor, 'v')])
                    neighbor_ranges.extend([range(len(spacing_ranges[index])-1), 
                                         range(len(velocity_ranges[index])-1)])
            

            for idx_combination in itertools.product(*neighbor_ranges):
                current_bounds = base_bounds.copy()
                
                # Apply each index to the corresponding neighbor and state
                for (neighbor_idx, state_type), grid_idx in zip(neighbor_indices, idx_combination):
                    if state_type == 's':
                        current_bounds[2*neighbor_idx] = [
                            spacing_ranges[index][grid_idx],
                            spacing_ranges[index][grid_idx + 1]
                        ]
                    else:  # state_type == 'v'
                        current_bounds[2*neighbor_idx + 1] = [
                            velocity_ranges[index][grid_idx],
                            velocity_ranges[index][grid_idx + 1]
                        ]
                yield current_bounds

        idx = 0
        # Verify each combination of neighbor states
        for state_bounds in generate_neighbor_bounds(agent_id-1):
            print('agent_id', agent_id, 'idx', idx)
            ans = query.check_descent(state_bounds)
            idx += 1
            if isinstance(ans, list) and len(ans) > 1:
                vals_found.append(ans)
                val_ranges.append(state_bounds)
            elif ans[0] == -1:
                failed_vals.append(ans)

        # Determine verification result for this agent
        if len(vals_found) > 0:
            result = f"Agent {agent_id}: Failed (found {len(vals_found)} counterexamples)"
        elif len(failed_vals) > 0:
            result = f"Agent {agent_id}: Inconclusive ({len(failed_vals)} queries failed)"
        else:
            result = f"Agent {agent_id}: Succeeded"

        results.append(result)  
    
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

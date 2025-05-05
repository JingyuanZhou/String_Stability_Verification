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

    def check_descent(self, input_bounds, epsilon=0.0001, useMILP=True):
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
            omega_error_range.append(np.linspace(-0.5, 0.5, split_nums[k_]))  # Frequency error range (rad/s)
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

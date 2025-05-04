import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import single_actor
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm

def integrated_gradients(model, input_tensor, baseline, steps=50):
    """
    Compute the absolute Integrated Gradients (IG) for a given input relative to a baseline.
    
    input_tensor and baseline are 1D tensors (of length = 2*m).
    """
    scaled_inputs = [baseline + (float(i)/steps)*(input_tensor - baseline)
                     for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs)
    model.eval()
    outputs = model(scaled_inputs)
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = outputs[:, 0]
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=scaled_inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=False,
        retain_graph=False
    )[0]
    avg_grads = grads.mean(dim=0)
    ig = (input_tensor - baseline) * avg_grads
    return ig.abs()

def cal_expected_agent_importances(model, num_agents, sample_bounds, num_samples=10000, steps=50):
    """
    Compute the expected (average) absolute IG for each agent.
    
    Each agent is represented by 2 state values.
    sample_bounds: list of length num_agents, each element is a tuple (lb, ub)
                   applied uniformly to both dimensions of that agent.
    Returns: a tensor of length num_agents.
    """
    total_importances = torch.zeros(num_agents)
    # Construct the overall baseline: for each agent, use fixed baseline (e.g., (20,15)).
    baseline_agent = (20, 15)
    baseline = torch.FloatTensor(baseline_agent * num_agents)
    
    for _ in tqdm(range(num_samples), desc="Calculating Expected Agent Importances"):
        sample = []
        for i in range(num_agents):
            lb, ub = sample_bounds[i]
            # For each agent, sample a 2D vector from [lb, ub] for both dimensions.
            sample.extend([random.uniform(lb, ub), random.uniform(lb, ub)])
        sample_x = torch.FloatTensor(sample).requires_grad_(True)
        ig = integrated_gradients(model, sample_x, baseline, steps=steps)  # length = 2*num_agents
        # For each agent, aggregate the two IG values (here, take the average).
        for i in range(num_agents):
            agent_importance = (ig[2*i] + ig[2*i+1]) / 2.0
            total_importances[i] += agent_importance
    expected_A = total_importances / num_samples
    return expected_A

def evaluate_individual(individual, model, sample_bounds, baseline, C, epsilon, penalty_factor, num_MC_samples=20):
    """
    Evaluate a candidate agent-selection mask (binary vector of length num_agents)
    over a continuous domain.
    
    For each Monte Carlo sample, each agent's 2D state is sampled uniformly from the given bound.
    The pruned state for an agent is:
      tilde_x^agent = baseline_agent + (x^agent - baseline_agent) * z,
    where z is the decision (0 or 1) for that agent.
    
    The fitness is the sum of the weighted cost (∑ C_i * z_i) plus the average penalty
    for the output deviation |F(tilde_x) - F(x)| exceeding epsilon.
    """
    m = len(individual)  # number of agents
    cost = sum(C[i] * individual[i] for i in range(m))
    total_penalty = 0.0
    
    for _ in range(num_MC_samples):
        sample = []
        for i in range(m):
            lb, ub = sample_bounds[i]
            # Sample 2D state for agent i from [lb, ub] for both dimensions.
            sample.extend([random.uniform(lb, ub), random.uniform(lb, ub)])
        x_sample = torch.FloatTensor(sample)
        z_tensor = torch.tensor(individual, dtype=torch.float32)  # shape: [m]
        # Build pruned input: for each agent, if z_i==1, keep the sample; else use baseline.
        baseline_agent = list(baseline.numpy()[:2])  # assumes same for all agents; here baseline is [20,15,20,15,...]
        tilde_sample = []
        for i in range(m):
            if individual[i] == 1:
                tilde_sample.extend(x_sample[2*i:2*i+2].tolist())
            else:
                tilde_sample.extend(baseline[2*i:2*i+2].tolist())
        tilde_x = torch.FloatTensor(tilde_sample)
        
        with torch.no_grad():
            F_sample = model(x_sample.unsqueeze(0)).item()
            F_tilde = model(tilde_x.unsqueeze(0)).item()
        diff = abs(F_tilde - F_sample)
        total_penalty += penalty_factor * max(0.0, diff - epsilon)
    
    avg_penalty = total_penalty / num_MC_samples
    fitness = cost + avg_penalty
    return (fitness,)

def run_deap_ga(model, sample_bounds, baseline, C, epsilon, penalty_factor,
                population_size=50, generations=100, cxpb=0.8, mutpb=0.1, num_MC_samples=20):
    """
    Run a Genetic Algorithm using DEAP to find an agent-selection mask z (length = num_agents)
    that is robust over the entire input domain.
    
    The evaluation is performed via Monte Carlo sampling over the domain defined by sample_bounds.
    """
    num_agents = len(baseline) // 2  # each agent has 2 states
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_agents)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, model=model, sample_bounds=sample_bounds,
                     baseline=baseline, C=C, epsilon=epsilon, penalty_factor=penalty_factor, num_MC_samples=num_MC_samples)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits)/len(fits))
    stats.register("min", min)
    
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations,
                         stats=stats, halloffame=hof, verbose=False)
    
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]
    
    # For reporting, sample one instance from sample_bounds.
    sample = []
    for i in range(num_agents):
        lb, ub = sample_bounds[i]
        sample.extend([random.uniform(lb, ub), random.uniform(lb, ub)])
    x_sample = torch.FloatTensor(sample)
    
    best_z_tensor = torch.tensor(best_individual, dtype=torch.float32)
    # Construct pruned input: for each agent, if z==1 keep sample, else baseline.
    best_pruned = []
    for i in range(num_agents):
        if best_individual[i] == 1:
            best_pruned.extend(x_sample[2*i:2*i+2].tolist())
        else:
            best_pruned.extend(baseline[2*i:2*i+2].tolist())
    best_tilde_x = torch.FloatTensor(best_pruned)
    
    with torch.no_grad():
        F_sample = model(x_sample.unsqueeze(0)).item()
        F_tilde = model(best_tilde_x.unsqueeze(0)).item()
    best_F_diff = abs(F_tilde - F_sample)
    
    return best_individual, best_fitness, best_tilde_x, best_F_diff

if __name__ == "__main__":
    # Number of agents (each agent has 2 state values).
    num_agents = 5
    input_dim = num_agents * 2  # total number of state values.
    
    # For each agent, define sample bounds as a tuple (lb, ub). For simplicity, we use the same bound for both dimensions.
    sample_bounds = [(10, 30)] * num_agents
    
    # Define the overall baseline by concatenating each agent's baseline.
    # For instance, for each agent, baseline is (20,15)
    baseline = torch.FloatTensor([20, 15] * num_agents)
    
    # Load the actor network (using your single_actor) and pre-trained weights.
    actor = single_actor(state_dim=input_dim, control_dim=1)
    state_dict = torch.load("pre_train_model/sac_platoon_99_actor.pth")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("trunk", "network")
        if "network.4.weight" in new_key:
            v = v[:1, :]  # adjust if necessary
        elif "network.4.bias" in new_key:
            v = v[:1]
        new_state_dict[new_key] = v
    actor.load_state_dict(new_state_dict)
    actor.eval()
    
    # Compute expected agent importances via Monte Carlo sampling.
    print("Calculating expected agent importances...")
    expected_A = cal_expected_agent_importances(actor, num_agents, sample_bounds, num_samples=10000, steps=50)
    print("Expected agent importances:", expected_A.tolist())
    
    # Compute cost coefficients: C_i = 1/(A_i + δ)
    delta_val = 1e-4
    C = [1.0 / (expected_A[i].item() + delta_val) for i in range(num_agents)]
    
    # Tolerance and penalty settings.
    epsilon = 0.1
    penalty_factor = 6.0
    
    # Run GA to obtain a robust agent-selection mask.
    best_z, best_obj, best_tilde_x, best_F_diff = run_deap_ga(
        actor, sample_bounds, baseline, C, epsilon, penalty_factor,
        population_size=50, generations=100, cxpb=0.8, mutpb=0.1, num_MC_samples=20)
    
    print("\nOptimal agent-selection (GA result):")
    for i in range(num_agents):
        print(f"Agent {i}: selected = {best_z[i]}")
    print("Representative pruned agent state (tilde_x):", best_tilde_x.tolist())
    print("Objective value =", best_obj)
    print("Representative output difference |F(tilde_x)-F(x)| =", best_F_diff)

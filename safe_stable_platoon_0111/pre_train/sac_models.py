import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math

from torch import distributions as pyd
from torch.autograd import Variable, grad

def orthogonal_init(layer):
    """正交初始化，只对Linear层进行初始化"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        nn.init.constant_(layer.bias, 0)
    return layer

def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    """创建MLP网络"""
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    
    # 对每一层进行初始化
    trunk = nn.Sequential(*mods)
    for m in trunk.modules():
        orthogonal_init(m)
    
    return trunk

class SingleQCritic(nn.Module):
    """Single Q-value critic with optional centralized functionality"""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args, centralized=False, num_agents=3):
        super().__init__()
        
        self.centralized = centralized
        self.num_agents = num_agents
        self.multi_agent = num_agents > 1
        
        if centralized and self.multi_agent:
            # Centralized critic gets global state and all agent actions
            # The states already contain information for all agents
            input_dim = obs_dim + action_dim
            # Use larger network for centralized critic
            critic_hidden_dim = hidden_dim * 2
        else:
            # Regular critic only gets individual state and action
            input_dim = obs_dim + action_dim
            critic_hidden_dim = hidden_dim
            
        self.Q = mlp(input_dim, critic_hidden_dim, 1, hidden_depth + (1 if centralized else 0))
        self.args = args

    def forward(self, obs, action, both=False):
        # For centralized critic, obs already contains global information
        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)
        return q

class DoubleQCritic(nn.Module):
    """Double Q-value critic with optional centralized functionality"""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args, centralized=False, num_agents=3):
        super().__init__()
        
        self.centralized = centralized
        self.num_agents = num_agents
        self.multi_agent = num_agents > 1
        
        if centralized and self.multi_agent:
            # Centralized critic gets global state and all agent actions
            # The states already contain information for all agents
            input_dim = obs_dim + action_dim
            # Use larger network for centralized critic
            critic_hidden_dim = hidden_dim * 2
        else:
            # Regular critic only gets individual state and action
            input_dim = obs_dim + action_dim
            critic_hidden_dim = hidden_dim
        
        self.Q1 = mlp(input_dim, critic_hidden_dim, 1, hidden_depth + (1 if centralized else 0))
        self.Q2 = mlp(input_dim, critic_hidden_dim, 1, hidden_depth + (1 if centralized else 0))
        self.args = args

    def forward(self, obs, action, both=False):
        # For centralized critic, obs already contains global information
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if both:
            return q1, q2
        return torch.min(q1, q2)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()
        
        self.log_std_min, self.log_std_max = log_std_bounds
        self.log_std_bounds = log_std_bounds
        self.action_dim = action_dim  # Store action dimension for multi-agent handling
        
        # Increase network capacity for multi-agent scenarios
        if action_dim > 1:
            # For multi-agent, use a larger network
            hidden_dim = max(hidden_dim, 64)
            
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
    
    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        
        # Calculate log_prob for each action dimension
        log_prob = dist.log_prob(action)
        
        # Always ensure log_prob has shape [batch_size, action_dim]
        if len(log_prob.shape) == 1:
            # If log_prob is [batch_size], reshape to [batch_size, 1]
            log_prob = log_prob.unsqueeze(-1)
            
        return action, log_prob, dist.mean

'''
    def sample(self, obs):
        dist = self.forward(obs)
        action = torch.tanh(dist.rsample())
        
        # 计算log_prob，添加数值稳定性
        log_prob = dist.log_prob(action)
        
        # 修正log_prob
        # log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-6))
        # log_prob = torch.clamp(log_prob, min=-20, max=20)
        # log_prob = log_prob.sum(-1, keepdim=True)
        
        mean = torch.tanh(dist.mean)
        
        return action, log_prob, mean
'''

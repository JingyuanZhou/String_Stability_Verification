import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_models import DoubleQCritic, DiagGaussianActor, SingleQCritic
#from utils.iq_learn import get_concat_samples, iq_loss
import torch.nn as nn

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class SAC(nn.Module):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        super().__init__()
        
        self.args = args
        self.device = args['device']
        self.gamma = args['gamma']
        self.batch_size = batch_size
        self.action_range = action_range
        self.critic_tau = args['critic_tau']
        self.action_dim = action_dim  # Store action dimension for multi-agent handling
        
        # Determine if we're in multi-agent mode (more than one CAV)
        self.multi_agent = action_dim > 1
        self.num_agents = action_dim if self.multi_agent else 1
        
        # Use centralized critic for multi-agent learning
        self.use_centralized_critic = self.multi_agent
        
        # 初始化网络
        if args['method'].get('single_q', False):
            self.critic = SingleQCritic(
                obs_dim, action_dim, args['hidden_dim'], 
                args['hidden_depth'], args,
                centralized=self.use_centralized_critic,
                num_agents=self.num_agents
            ).to(self.device)
            
            self.critic_target = SingleQCritic(
                obs_dim, action_dim, args['hidden_dim'], 
                args['hidden_depth'], args,
                centralized=self.use_centralized_critic,
                num_agents=self.num_agents
            ).to(self.device)
        else:
            self.critic = DoubleQCritic(
                obs_dim, action_dim, args['hidden_dim'], 
                args['hidden_depth'], args,
                centralized=self.use_centralized_critic,
                num_agents=self.num_agents
            ).to(self.device)
            
            self.critic_target = DoubleQCritic(
                obs_dim, action_dim, args['hidden_dim'], 
                args['hidden_depth'], args,
                centralized=self.use_centralized_critic,
                num_agents=self.num_agents
            ).to(self.device)
        
        # For multi-agent, create separate actors for each agent
        if self.multi_agent:
            self.actors = nn.ModuleList([
                DiagGaussianActor(
                    obs_dim, 1, args['hidden_dim'],  # Each actor outputs 1 action
                    args['hidden_depth'], args['log_std_bounds']
                ).to(self.device)
                for _ in range(self.num_agents)
            ])
            self.actor_optimizers = [
                Adam(actor.parameters(), lr=args['actor_lr'])
                for actor in self.actors
            ]
            
            # Separate alpha parameters for each agent
            self.target_entropies = [-1 for _ in range(self.num_agents)]  # Target entropy per agent
            self.log_alphas = [
                torch.tensor(np.log(args['init_temp']), requires_grad=True, device=self.device)
                for _ in range(self.num_agents)
            ]
            self.alphas = [args['init_temp'] for _ in range(self.num_agents)]
            self.log_alpha_optimizers = [
                Adam([self.log_alphas[i]], lr=args['alpha_lr'])
                for i in range(self.num_agents)
            ]
        else:
            # Single agent case
            self.actor = DiagGaussianActor(
                obs_dim, action_dim, args['hidden_dim'], 
                args['hidden_depth'], args['log_std_bounds']
            ).to(self.device)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=args['actor_lr'])
            self.target_entropy = -action_dim
            self.log_alpha = torch.tensor(np.log(args['init_temp']), requires_grad=True, device=self.device)
            self._alpha = args['init_temp']
            self.log_alpha_optimizer = Adam([self.log_alpha], lr=args['alpha_lr'])

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args['critic_lr'])
        
    @property
    def alpha(self):
        if self.multi_agent:
            return self.alphas
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        if self.multi_agent:
            if isinstance(value, list):
                self.alphas = value
            else:
                self.alphas = [value] * self.num_agents
        else:
            self._alpha = value

    def choose_action(self, state, sample=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            
            if self.multi_agent:
                # For multi-agent, get action from each actor
                actions = []
                for actor in self.actors:
                    dist = actor(state)
                    action = dist.sample() if sample else dist.mean
                    actions.append(action)
                
                # Combine actions from all actors
                combined_action = torch.cat(actions, dim=1)
                return combined_action.detach().cpu().numpy()[0]
            else:
                # Single agent case
                dist = self.actor(state)
                action = dist.sample() if sample else dist.mean
                return action.detach().cpu().numpy()[0]

    def update(self, memory, writer, step):
        batch = memory.sample(self.batch_size)
        obs, next_obs, action, reward, done = batch
        
        # 将数据转移到正确的设备
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        
        # 更新critic with centralized learning
        with torch.no_grad():
            if self.multi_agent:
                # Get next action from each actor and combine
                next_actions = []
                next_log_probs = []
                
                for i, actor in enumerate(self.actors):
                    next_action, next_log_prob, _ = actor.sample(next_obs)
                    next_actions.append(next_action)
                    next_log_probs.append(next_log_prob)
                    
                # Combine actions and log_probs
                next_action = torch.cat(next_actions, dim=1)
                next_log_prob = torch.cat(next_log_probs, dim=1).sum(dim=1, keepdim=True)
            else:
                next_action, next_log_prob, _ = self.actor.sample(next_obs)
                if self.action_dim > 1:
                    next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
            
            # Get target Q value
            if self.args['method'].get('single_q', False):
                target_Q = self.critic_target(next_obs, next_action)
                if self.multi_agent:
                    # Use average of alphas for multi-agent
                    avg_alpha = sum(self.alphas) / len(self.alphas)
                    target_V = target_Q - avg_alpha * next_log_prob
                else:
                    target_V = target_Q - self._alpha * next_log_prob
            else:
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action, both=True)
                if self.multi_agent:
                    # Use average of alphas for multi-agent
                    avg_alpha = sum(self.alphas) / len(self.alphas)
                    target_V = torch.min(target_Q1, target_Q2) - avg_alpha * next_log_prob
                else:
                    target_V = torch.min(target_Q1, target_Q2) - self._alpha * next_log_prob
                    
            target_Q = reward + (1 - done) * self.gamma * target_V
        
        # Update centralized critic
        if self.args['method'].get('single_q', False):
            current_Q = self.critic(obs, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
        else:
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新actor with decentralized execution
        if self.multi_agent:
            actor_losses = []
            alpha_losses = []
            
            # Split action into per-agent actions
            actions = torch.split(action, 1, dim=1)
            
            for i, actor in enumerate(self.actors):
                # Get new action for this agent
                action_new, log_prob, _ = actor.sample(obs)
                
                # Combine this agent's new action with other agents' original actions for critic evaluation
                combined_action = list(actions)  # Make a copy of the original actions
                combined_action[i] = action_new  # Replace this agent's action
                combined_action = torch.cat(combined_action, dim=1)
                
                # Get Q value from centralized critic
                if self.args['method'].get('single_q', False):
                    Q = self.critic(obs, combined_action)
                else:
                    Q1, Q2 = self.critic(obs, combined_action, both=True)
                    Q = torch.min(Q1, Q2)
                
                # Calculate actor loss
                actor_loss = (self.alphas[i] * log_prob - Q).mean()
                actor_losses.append(actor_loss.item())
                
                # Update actor
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()
                
                # Update alpha
                alpha_loss = -(self.log_alphas[i] * (log_prob.detach() + self.target_entropies[i])).mean()
                alpha_losses.append(alpha_loss.item())
                
                self.log_alpha_optimizers[i].zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizers[i].step()
                
                # Update alpha value
                self.alphas[i] = self.log_alphas[i].exp().item()
        else:
            # Single agent update
            action_new, log_prob, _ = self.actor.sample(obs)
            
            if self.action_dim > 1:
                log_prob = log_prob.sum(dim=1, keepdim=True)
                
            if self.args['method'].get('single_q', False):
                Q = self.critic(obs, action_new)
            else:
                Q1, Q2 = self.critic(obs, action_new, both=True)
                Q = torch.min(Q1, Q2)
                
            actor_loss = (self._alpha * log_prob - Q).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update alpha
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            
            # Update alpha value
            self._alpha = self.log_alpha.exp().item()
        
        # 更新目标网络
        self._soft_update_target()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': np.mean(actor_losses) if self.multi_agent else actor_loss.item(),
            'alpha_loss': np.mean(alpha_losses) if self.multi_agent else alpha_loss.item(),
            'alpha': np.mean(self.alphas) if self.multi_agent else self._alpha,
            'q1': current_Q1.mean().item() if not self.args['method'].get('single_q', False) else 0,
            'q2': current_Q2.mean().item() if not self.args['method'].get('single_q', False) else 0
        }

    def save(self, path):
        if self.multi_agent:
            for i, actor in enumerate(self.actors):
                torch.save(actor.state_dict(), f"{path}_actor_{i}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load(self, path):
        if self.multi_agent:
            for i, actor in enumerate(self.actors):
                actor.load_state_dict(torch.load(f"{path}_actor_{i}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
        
    def _soft_update_target(self):
        """软更新目标网络"""
        soft_update(self.critic_target, self.critic, self.critic_tau)

    def iq_update(self, policy_batch, expert_batch, logger, step):
        """IQ-Learn的主更新函数"""
        # policy_batch和expert_batch已经是采样好的数据
        # 不需要再次采样
        
        # 更新critic
        losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)
        
        # 更新actor和alpha
        if step % self.args['train']['actor_update_freq'] == 0:
            # 使用策略和专家的观测
            if self.args['method'].get('only_expert_states', False):
                obs = expert_batch[0]
            else:
                policy_obs = policy_batch[0]
                expert_obs = expert_batch[0]
                obs = torch.cat([policy_obs, expert_obs], dim=0)
            
            actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
            losses.update(actor_alpha_losses)
        
        # 更新目标网络
        if step % self.args['train']['target_update_freq'] == 0:
            self._soft_update_target()
            
        return losses

    def iq_update_critic(self, policy_batch, expert_batch, logger, step):
        """更新critic网络"""
        policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
        expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
        
        # 处理只使用专家状态的情况
        if self.args['method'].get('only_expert_states', False):
            expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done
        
        # 合并数据
        batch = get_concat_samples(policy_batch, expert_batch)
        obs, next_obs, action = batch[0:3]
        
        # 计算当前值和目标值
        current_v = self.getV(obs)
        if self.args['train'].get('use_target', False):
            with torch.no_grad():
                next_v = self.get_targetV(next_obs)
        else:
            next_v = self.getV(next_obs)
        
        # 使用双Q网络
        if self.args['method'].get('single_q', False):
            current_Q = self.critic(obs, action)
            q_loss, loss_dict = self.iq_learn_loss(current_Q, current_v, next_v, batch)
            critic_loss = q_loss
        else:
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
            q1_loss, loss_dict1 = self.iq_learn_loss(current_Q1, current_v, next_v, batch)
            q2_loss, loss_dict2 = self.iq_learn_loss(current_Q2, current_v, next_v, batch)
            critic_loss = 0.5 * (q1_loss + q2_loss)
            loss_dict = {k: 0.5 * (loss_dict1[k] + loss_dict2[k]) for k in loss_dict1.keys()}
        
        # 优化critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return loss_dict

    def iq_learn_loss(self, current_Q, current_v, next_v, batch):
        """计算IQ-Learn损失"""
        obs, next_obs, action, reward, done, is_expert = batch
        
        # 确保is_expert是布尔类型
        is_expert = is_expert.bool()
        
        # 计算目标值
        y = (1 - done) * self.args['gamma'] * next_v
        
        # 计算专家数据的奖励
        reward = (current_Q - y)[is_expert]
        phi_grad = 1/(1-reward)**2
        loss = -(phi_grad*reward).mean()
        # loss = -(reward).mean()

        # 计算值函数损失
        if self.args['method'].get('only_expert_states', False):
            value_loss = (current_v - y)[is_expert].mean()
        else:
            value_loss = (current_v - y).mean()
        loss += value_loss
        
        # χ2散度损失
        chi2_loss = 1/(4 * self.args['method']['alpha']) * (reward**2).mean()
        if self.args['method'].get('only_expert_states', False) == False and self.args['method'].get('use_chi2', False):
            loss += chi2_loss
        
        return -loss, {
            'iq_loss': loss.item(),
            'value_loss': value_loss.item(),
            'chi2_loss': chi2_loss.item()
        }

    def update_actor_and_alpha(self, obs, logger, step):
        """更新actor和alpha"""
        # 计算actor损失
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        q_value = self.critic(obs, action)
        actor_loss = (self.alpha * log_prob - q_value).mean()
        
        # 优化actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        return {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }

    def getV(self, obs):
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs)
            current_Q = self.critic(obs, action)
            current_V = current_Q - self.alpha * log_prob
        return current_V
    
    def get_targetV(self, obs):
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs)
            target_Q = self.critic_target(obs, action)
            target_V = target_Q - self.alpha * log_prob
        return target_V
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
        
        # 初始化网络
        if args['method'].get('single_q', False):
            self.critic = SingleQCritic(obs_dim, action_dim, args['hidden_dim'], 
                                     args['hidden_depth'], args).to(self.device)
            self.critic_target = SingleQCritic(obs_dim, action_dim, args['hidden_dim'], 
                                              args['hidden_depth'], args).to(self.device)
        else:
            self.critic = DoubleQCritic(obs_dim, action_dim, args['hidden_dim'], 
                                  args['hidden_depth'], args).to(self.device)
            self.critic_target = DoubleQCritic(obs_dim, action_dim, args['hidden_dim'], 
                                         args['hidden_depth'], args).to(self.device)
        
        self.actor = DiagGaussianActor(obs_dim, action_dim, args['hidden_dim'], 
                                     args['hidden_depth'], args['log_std_bounds']).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 设置优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args['actor_lr'])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args['critic_lr'])
        
        # 设置温度参数alpha
        self.target_entropy = -action_dim  # 目标熵
        self.log_alpha = torch.tensor(np.log(args['init_temp'])).to(self.device)
        self.log_alpha.requires_grad = True
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self._alpha = args['init_temp']  # 初始温度
        self.log_alpha_optimizer = Adam([self.log_alpha], lr=args['alpha_lr'])
        
        # 现在可以使用register_buffer了
        # self.register_buffer('alpha', torch.tensor(self._alpha))
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    def choose_action(self, state, sample=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
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
        
        # 更新critic
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            if self.args['method'].get('single_q', False):
                target_Q = self.critic_target(next_obs, next_action)
                target_V = target_Q - self.alpha * next_log_prob
            else:
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action, both=True)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V
        
        if self.args['method'].get('single_q', False):
            current_Q = self.critic(obs, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
        else:
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新actor
        action_new, log_prob, _ = self.actor.sample(obs)
        if self.args['method'].get('single_q', False):
            Q = self.critic(obs, action_new)
        else:
            Q1, Q2 = self.critic(obs, action_new, both=True)
            Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_prob - Q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新alpha
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        # 更新alpha值
        self._alpha = self.log_alpha.exp().item()
        
        # 更新目标网络
        self._soft_update_target()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self._alpha,
            'q1': current_Q1.mean().item(),
            'q2': current_Q2.mean().item()
        }

    def save(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor"))
        self.critic.load_state_dict(torch.load(f"{path}_critic"))

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

    def _soft_update_target(self):
        """软更新目标网络"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)

        
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
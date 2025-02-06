import os
import random
import time
import datetime
from collections import deque
from itertools import count

import numpy as np
import torch
from tensorboardX import SummaryWriter
import torch.optim as optim

from env import PlatoonEnv
from sac import SAC
from memory import Memory

def train_expert():
    # 基础配置
    args = {
        'seed': 0,
        'device': 'cpu',
        
        # 环境参数
        'env': {
            'num_vehicles': 5,
            'dt': 0.1,
            'cav_index': [1,3],
            'select_scenario': 0,
        },
        
        # SAC参数
        'agent': {
            'gamma': 0.99,
            'critic_tau': 0.005,
            'init_temp': 0.1,
            'hidden_dim': 30,
            'hidden_depth': 2,
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'alpha_lr': 1e-4,
            'batch_size': 64,
            'log_std_bounds': [-10, 2],
            'device': 'cpu',
            'method': {
                'type': 'iq',
                'alpha': 10,  # χ2散度的系数
                'only_expert_states': False,
                'single_q': False,
                'use_chi2': True
            }
        },
        
        # 训练参数
        'train': {
            'replay_mem': int(1e6),
            'initial_mem': int(1e4),
            'max_steps': 2000,
            'eval_interval': 1000,
            'num_eval_episodes': 5,
            'log_interval': 100,
            'save_interval': 50,
            'num_episodes': 100
        },
        
        # 日志参数
        'log': {
            'log_dir': 'logs',
            'exp_name': 'sac_expert'
        }
    }

    # 设置随机种子
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])

    # 创建环境
    env = PlatoonEnv(**args['env'])
    eval_env = PlatoonEnv(**args['env'])

    # 创建智能体
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    
    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=action_range,
        batch_size=args['agent']['batch_size'],
        args=args['agent']
    )

    # 创建学习率调度器
    actor_scheduler = optim.lr_scheduler.StepLR(agent.actor_optimizer, step_size=1000, gamma=0.96)
    critic_scheduler = optim.lr_scheduler.StepLR(agent.critic_optimizer, step_size=1000, gamma=0.96)
    alpha_scheduler = optim.lr_scheduler.StepLR(agent.log_alpha_optimizer, step_size=1000, gamma=0.96)

    # 创建经验回放内存
    memory = Memory(args['train']['replay_mem'], args['seed'])

    # 设置日志
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args['log']['log_dir'], args['log']['exp_name'], ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')

    # 训练跟踪
    rewards_window = deque(maxlen=100)
    best_eval_returns = -np.inf
    steps = 0
    learn_steps = 0
    begin_learn = False
    initial_memory = args['train']['initial_mem']
    num_episodes = args['train']['num_episodes']

    if args['env']['select_scenario'] == 1:
        num_episodes = 100#env.NGSIM_episodes - 1
        print(f'Number of episodes: {num_episodes}')

    for epoch in range(num_episodes):
        state = env.reset()
        env.episiode_id = epoch
        episode_reward = 0
        done = False
        episode_steps = 0

        start_time = time.time()

        while not done and episode_steps < args['train']['max_steps']:
            if steps < initial_memory:
                # 随机采样动作来填充经验回放
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state, sample=True)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            episode_steps += 1
            # 存储经验
            memory.add((state, next_state, action, reward, done))
            # 评估
            if learn_steps % args['train']['eval_interval'] == 0:
                eval_returns, eval_steps = evaluate(agent, eval_env, num_episodes=5)
                returns = np.mean(eval_returns)
                learn_steps += 1  # 防止重复评估
                
                writer.add_scalar('eval/episode_reward', returns, learn_steps)
                writer.add_scalar('eval/episode', epoch, learn_steps)
                
                if returns > best_eval_returns:
                    best_eval_returns = returns
                    save(agent, -1, args, output_dir='pre_train_model')

            # 训练
            if memory.size() > initial_memory:
                if not begin_learn:
                    print('Begin learning!')
                    begin_learn = True

                learn_steps += 1
                losses = agent.update(memory, writer, learn_steps)

                # 更新学习率
                actor_scheduler.step()
                critic_scheduler.step()
                alpha_scheduler.step()

                if learn_steps % args['train']['log_interval'] == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)
                    save(agent, epoch, args, output_dir='pre_train_model')
            state = next_state

        # 记录episode信息
        rewards_window.append(episode_reward)
        writer.add_scalar('train/episode', epoch, learn_steps)
        writer.add_scalar('train/episode_reward', episode_reward, learn_steps)
        writer.add_scalar('train/duration', time.time() - start_time, learn_steps)
        
        print(f'Episode {epoch}: reward={episode_reward:.2f}, steps={episode_steps}, learning_rate={actor_scheduler.get_last_lr()[0]:.2e}')

        # 定期保存模型
        #if epoch % args['train']['save_interval'] == 0:
        #    save(agent, epoch, args)

def evaluate(agent, env, num_episodes=1):
    """评估函数"""
    returns = []
    steps = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, sample=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
        returns.append(episode_reward)
        steps.append(episode_steps)
    
    return returns, steps

def save(agent, epoch, args, output_dir='results'):
    """保存模型"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    agent.save(f'{output_dir}/sac_platoon_{epoch}')

if __name__ == "__main__":
    train_expert()
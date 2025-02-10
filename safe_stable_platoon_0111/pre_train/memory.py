import numpy as np
from collections import deque
import random
import torch
import pickle
import os

class Memory:
    def __init__(self, capacity, seed):
        """初始化Memory类
        
        Args:
            capacity (int): 经验回放的容量
            seed (int): 随机种子
        """
        np.random.seed(seed)
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition):
        """添加一条经验到内存中
        
        Args:
            transition: tuple (state, next_state, action, reward, done)
        """
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """随机采样一个批次的经验
        
        Args:
            batch_size (int): 批次大小
            
        Returns:
            tuple: (states, next_states, actions, rewards, dones)
            每个元素都是torch.Tensor
        """
        indices = np.random.randint(len(self.buffer), size=batch_size)
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        
        for idx in indices:
            state, next_state, action, reward, done = self.buffer[idx]
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards).reshape(-1, 1)),
            torch.FloatTensor(np.array(dones).reshape(-1, 1))
        )
    
    def save(self, filename):
        """保存经验回放数据到文件
        
        Args:
            filename (str): 保存路径
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Saved {len(self.buffer)} transitions to {filename}")
    
    def load(self, filename, num_trajs=None, sample_freq=1):
        """从文件加载经验回放数据
        
        Args:
            filename (str): 加载路径
            num_trajs (int, optional): 加载的轨迹数量，None表示全部加载
            sample_freq (int): 采样频率，每隔多少步采样一次
        """
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist!")
            return
            
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        # 如果指定了轨迹数量，按轨迹切分数据
        if num_trajs is not None:
            # 找到轨迹的结束点
            traj_ends = [i for i, (_, _, _, _, done) in enumerate(data) if done]
            if len(traj_ends) == 0:
                print("Warning: No complete trajectories found in data!")
                return
                
            # 只保留指定数量的轨迹
            if num_trajs < len(traj_ends):
                last_idx = traj_ends[num_trajs-1]
                data = data[:last_idx+1]
        
        # 按采样频率加载数据
        self.buffer.clear()
        for i in range(0, len(data), sample_freq):
            self.buffer.append(data[i])
            
        print(f"Loaded {len(self.buffer)} transitions from {filename}")
    
    def size(self):
        """返回当前内存中的经验数量"""
        return len(self.buffer)
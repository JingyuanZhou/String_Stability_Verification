from typing import Tuple
import numpy as np
import gym
from gym import spaces
from numpy import random
import pandas as pd

class PlatoonEnv(gym.Env):
    def __init__(self, num_vehicles = 6, dt = 0.1, cav_index = [2], select_scenario = 0):
        super().__init__()
        
        # 基本参数
        self.num_vehicles = num_vehicles
        self.dt = dt
        self.cav_index = cav_index
        self.select_scenario = select_scenario
        self.steps = 0
        self.max_steps = 2000
        
        # 动作和观察空间
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(2 * 3,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-5,
            high=5, 
            shape=(1,),
            dtype=np.float32
        )

        # 状态变量
        self.spacing = np.zeros(num_vehicles)
        self.velocity = np.zeros(num_vehicles)
        self.position = np.zeros(num_vehicles)
        self.acceleration = np.zeros(num_vehicles)
        
        # 车辆动力学参数
        self.v0 = 15  # 平衡速度
        self.s0 = 20  # 平衡车距
        self.a_max = 5
        self.a_min = -5
        self.alpha = 0.6
        self.beta = 0.9
        self.s_go = 35
        self.s_st = 5
        self.v_max = 30
        
        # 头车抖动参数
        if self.select_scenario == 0:
            self.disturbance_amplitude = 5  # 抖动幅度
        else:
            self.use_max_id_vehicle = True
            # using NGSIM for the head vehicle velocity
            self.car_following_data = pd.read_csv("C:\\Users\\jyzho\\Desktop\\code\\NGSIM_car_following\\data\\useful_traj_pairs.csv")
            # 筛掉长度小于1000的轨迹（对于每个vehicle，使用后groupby），没有length这个列，使用groupby的size
            self.car_following_data = self.car_following_data.groupby('Vehicle_ID').filter(lambda x: len(x) > 1000)
            # 找到轨迹长度最长的vehicle id
            self.vehicle_id_max = self.car_following_data.groupby('Vehicle_ID').size().idxmax()
            print(f"vehicle_id_max: {self.vehicle_id_max}")
            print(f"length of vehicle_id_max: {len(self.car_following_data[self.car_following_data['Vehicle_ID'] == self.vehicle_id_max])}")
            self.max_length = len(self.car_following_data[self.car_following_data['Vehicle_ID'] == self.vehicle_id_max])
            self.vehicle_id = self.car_following_data['Vehicle_ID'].unique()
            self.NGSIM_episodes = len(self.vehicle_id)

        self.episiode_id = 0

    def reset(self):
        self.steps = 0
        
        # 重置状态
        self.spacing.fill(self.s0)
        self.velocity.fill(self.v0)
        self.acceleration.fill(0)
        
        # 设置初始位置
        for i in range(self.num_vehicles):
            self.position[i] = (self.num_vehicles - i - 1) * self.s0
            
        if self.select_scenario == 1:
            if self.use_max_id_vehicle:
                self.veh_velocity_traj = self.car_following_data[self.car_following_data['Vehicle_ID'] == self.vehicle_id_max]['v_Vel'].values
            else:
                self.veh_velocity_traj = self.car_following_data[self.car_following_data['Vehicle_ID'] == self.vehicle_id[self.episiode_id]]['v_Vel'].values
            self.length_veh_velocity_traj = len(self.veh_velocity_traj)
        
        return self.get_obs()

    def step(self, action, vehicle_id = None):
        # 更新控制输入
        self.acceleration[self.cav_index[0]] = action[0]
        
        # 头车随机抖动 (每个step都有)
        if self.select_scenario == 0:
            self.acceleration[0] = self.disturbance_amplitude * (2 * np.random.random() - 1)
        else:
            # using NGSIM for the head vehicle velocity
            self.velocity[0] = self.veh_velocity_traj[self.steps]
        
        # 更新环境
        self._update_vehicles()
        
        # 获取新状态和奖励
        next_obs = self.get_obs()
        reward = self._get_reward()

        if self.select_scenario == 1:
            done = (self.steps >= (self.length_veh_velocity_traj-1))
        else:
            done = (self.steps >= self.max_steps)
        
        self.steps += 1
        
        return next_obs, reward, done, {}
    
    def _update_vehicles(self):
        # 更新所有车辆状态
        for i in range(self.num_vehicles - 1, 0, -1):
            dv = self.velocity[i-1] - self.velocity[i]
            if i not in self.cav_index:
                # FVD update
                # calculate the desired acceleration
                cal_D = self.spacing[i]
                if cal_D > self.s_go:
                    cal_D = self.s_go
                elif cal_D < self.s_st:
                    cal_D = self.s_st

                self.acceleration[i] = self.alpha * (self.v_max/2*(1-np.cos(np.pi*(cal_D-self.s_st)/(self.s_go-self.s_st))) - self.velocity[i]) + self.beta * dv
                if self.acceleration[i] > self.a_max:
                    self.acceleration[i] = self.a_max
                elif self.acceleration[i] < self.a_min:
                    self.acceleration[i] = self.a_min

            
            # 更新速度和位置
            self.velocity[i] += self.acceleration[i] * self.dt
            self.position[i] += self.velocity[i] * self.dt
            self.spacing[i] += dv * self.dt

        
        # 更新头车
        if self.select_scenario == 0:
            self.velocity[0] += self.acceleration[0] * self.dt
            self.position[0] += self.velocity[0] * self.dt
        elif self.select_scenario == 1:
            self.velocity[0] = self.veh_velocity_traj[self.steps]
            self.position[0] += self.velocity[0] * self.dt
        
    def get_obs(self):
        return np.concatenate([
            self.velocity[self.cav_index[0]-1:self.cav_index[0]+2],
            self.spacing[self.cav_index[0]-1:self.cav_index[0]+2]
        ]).astype(np.float32)
    
    def get_states(self):
        return np.concatenate([
            self.velocity,
            self.spacing
        ]).astype(np.float32)
    
    def get_position(self):
        return self.position.astype(np.float32)

    def _get_reward(self):
        # 计算安全性奖励
        
        ttc = self.spacing[self.cav_index[0]] / (self.velocity[self.cav_index[0]-1] - self.velocity[self.cav_index[0]] + 1e-6)
        if 0 < ttc < 4:
            safety = np.log(ttc / 4)
        else:
            safety = 0
                
        # 计算效率奖励
        efficiency = 0
        for i in self.cav_index:
            if self.spacing[i]/self.velocity[i] > 2.5:  # 车距过大惩罚
                efficiency -= 2.5
                
        # 计算稳定性奖励
        stability = 0
        # calculate a decay weights for stability
        decay_weights = np.linspace(0.6, 0.1, self.num_vehicles - self.cav_index[0]+1)
        for i in range(self.cav_index[0], self.cav_index[0]+2):
            stability -= decay_weights[i - self.cav_index[0]] * (self.velocity[i] - self.velocity[i-1])**2

        fuel_consumption = -self.acceleration[self.cav_index[0]]**2

        # spacing equilibrium
        # print(self.spacing[self.cav_index[0]])
        spacing_equilibrium = -(self.spacing[self.cav_index[0]] - self.s0)**2
        
            
        reward_weights = [0.4, 0.3, 0.2, 0.1, 0.0]
        # print(f"reward_safety: {safety}, reward_efficiency: {efficiency}, reward_stability: {stability}, reward_fuel_consumption: {fuel_consumption}, reward_spacing_equilibrium: {spacing_equilibrium}")

        return safety * reward_weights[0] + efficiency * reward_weights[1] + stability * reward_weights[2] + fuel_consumption * reward_weights[3] + spacing_equilibrium * reward_weights[4]
    

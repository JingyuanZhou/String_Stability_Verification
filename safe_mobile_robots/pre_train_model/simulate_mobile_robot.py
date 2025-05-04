import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# ==================================================
# 1) 双积分系统动力学类
# ==================================================
class DoubleIntegratorSystem:
    """
    双积分模型:
      [ x_dot     ]   [ vx ]
      [ y_dot     ] = [ vy ]
      [ vx_dot    ]   [ u_x ]
      [ vy_dot    ]   [ u_y ]
    
    状态: [x, y, vx, vy]
    """
    def __init__(self, init_state, dt=0.1):
        """
        init_state: numpy 数组 shape (4,), 表示 [x0, y0, vx0, vy0]
        dt: 时间步长
        """
        self.state = np.array(init_state, dtype=float)
        self.dt = dt
    
    def update(self, control_input):
        """
        根据控制输入(加速度向量 [u_x, u_y])更新状态
        control_input: shape (2,)
        """
        x, y, vx, vy = self.state
        u_x, u_y = control_input

        # 欧拉积分
        x_new  = x  + vx * self.dt
        y_new  = y  + vy * self.dt
        vx_new = vx + u_x * self.dt
        vy_new = vy + u_y * self.dt

        self.state = np.array([x_new, y_new, vx_new, vy_new], dtype=float)
    
    def get_position(self):
        """返回当前 (x, y)"""
        return self.state[0:2]
    
    def get_velocity(self):
        """返回当前 (vx, vy)"""
        return self.state[2:4]
    
    def get_state(self):
        """返回 [x, y, vx, vy]"""
        return self.state.copy()


# ==================================================
# 2) 环境类: 障碍物 & 排斥势梯度
# ==================================================
class Environment:
    """
    环境类, 存放障碍物数据, 提供排斥势梯度计算。
    本例假设圆形障碍物, 每个障碍物用 (x_center, y_center, radius) 表示。
    """
    def __init__(self, obstacles, d0=3.0):
        """
        obstacles: list of tuples [(x_c, y_c, r), ...]
        d0: 安全距离阈值, 距离障碍物表面小于 d0 时才产生排斥势
        """
        self.obstacles = obstacles
        self.d0 = d0
    
    def compute_repulsive_gradient(self, position, beta):
        """
        计算所有障碍物对 position 的排斥势梯度之和
        position: shape (2,)
        beta: 排斥势系数
        """
        grad = np.zeros(2)
        p = np.array(position, dtype=float)

        for (x_c, y_c, r) in self.obstacles:
            center = np.array([x_c, y_c], dtype=float)
            dist_center = np.linalg.norm(p - center)
            dist_surface = dist_center - r  # 到障碍物边缘的距离

            # 若与障碍边缘距离 < d0, 则产生排斥势
            if dist_surface < self.d0:
                epsilon = 1e-6
                # U_rep ~ 0.5 * beta * (1/d - 1/d0)^2
                # grad(U_rep) ~ ...
                rep_factor = beta * (1/dist_surface - 1/self.d0)**2

                # 方向: (p - center)/dist_center
                direction = -(p - center) / dist_center
                grad += rep_factor * direction
        
        return grad


# ==================================================
# 3) 人工势场控制器 (Double Integrator)
# ==================================================
class APFControllerDoubleIntegrator:
    """
    双积分模型的人工势场控制器:
      u = -K_v * v - K_p * grad(U)
    其中:
      grad(U) = alpha*(p - p_goal) + grad_rep_from_env
    """
    def __init__(self, alpha=2.0, beta=6.0, Kp=2.0, Kv=1.0, environment=None):
        """
        alpha: 吸引势系数 (目标)
        beta:  排斥势系数 (障碍)
        Kp:    势场位置增益
        Kv:    速度阻尼增益
        environment: Environment 对象, 用于计算排斥势
        """
        self.alpha = alpha
        self.beta = beta
        self.Kp = Kp
        self.Kv = Kv
        self.env = environment  # 用于访问 compute_repulsive_gradient
    
    def get_control(self, position, velocity, goal):
        """
        计算加速度控制输入 u = [u_x, u_y].
        position: (2,) [x, y]
        velocity: (2,) [vx, vy]
        goal:     (2,) [x_goal, y_goal]
        """
        # 吸引势梯度: alpha * (p - p_goal)
        grad_att = self.alpha * (position - goal)

        # 排斥势梯度: 由环境对象计算
        grad_rep = np.zeros(2)
        if self.env is not None:
            grad_rep = self.env.compute_repulsive_gradient(position, self.beta)
        
        # 总势场梯度
        grad_U = grad_att + grad_rep

        # 控制律: u = -Kv*v - Kp*grad(U)
        control_acc = - self.Kv * velocity - self.Kp * grad_U
        return control_acc


# ==================================================
# 4) 可视化函数
# ==================================================
def visualize(trajectories, obstacles, goals):
    """
    画出障碍物(圆形)、轨迹和目标
    trajectories: list of lists, 每个智能体的轨迹(仅位置)
    obstacles: [(x_c, y_c, r), ...]
    goals: [(x_goal, y_goal), ...]
    """
    plt.figure(figsize=(8,6))

    # 障碍物
    for (x_c, y_c, r) in obstacles:
        circle = Circle((x_c, y_c), r, fill=True, alpha=0.3)
        plt.gca().add_patch(circle)

    # 轨迹
    n_agents = len(trajectories)
    for i in range(n_agents):
        traj = np.array(trajectories[i])
        plt.plot(traj[:,0], traj[:,1], label=f'Agent {i+1} path')
        # 起始点
        plt.scatter(traj[0,0], traj[0,1], marker='o')
        # 目标点
        plt.scatter(goals[i][0], goals[i][1], marker='x')

    plt.title("Multi-Agent Double Integrator with APF (Tuned)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


def animate_agents(trajectories, obstacles, goals, interval=50, save_animation=False):
    """
    创建智能体移动的动画
    
    参数:
    trajectories: list of lists, 每个智能体的轨迹(仅位置)
    obstacles: [(x_c, y_c, r), ...]
    goals: [(x_goal, y_goal), ...]
    interval: 帧之间的间隔时间(毫秒)
    save_animation: 是否保存动画为文件
    """
    n_agents = len(trajectories)
    # 找出所有轨迹中的最大长度
    max_frames = max(len(traj) for traj in trajectories)
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置坐标轴范围
    x_min = min(min(traj[i][0] for i in range(len(traj))) for traj in trajectories) - 2
    x_max = max(max(traj[i][0] for i in range(len(traj))) for traj in trajectories) + 2
    y_min = min(min(traj[i][1] for i in range(len(traj))) for traj in trajectories) - 2
    y_max = max(max(traj[i][1] for i in range(len(traj))) for traj in trajectories) + 2
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # 绘制障碍物
    for (x_c, y_c, r) in obstacles:
        circle = Circle((x_c, y_c), r, fill=True, alpha=0.3, color='red')
        ax.add_patch(circle)
    
    # 绘制目标点
    for i, goal in enumerate(goals):
        ax.scatter(goal[0], goal[1], marker='x', color=f'C{i}', s=100, label=f'Goal {i+1}')
    
    # 轨迹线(初始为空)
    lines = []
    # 智能体当前位置(标记)
    agents_points = []
    
    # 为每个智能体创建轨迹线和当前位置标记
    for i in range(n_agents):
        line, = ax.plot([], [], 'C'+str(i), alpha=0.5, label=f'Agent {i+1} path')
        lines.append(line)
        point, = ax.plot([], [], 'o', color=f'C{i}', markersize=8)
        agents_points.append(point)
    
    # 标题和图例
    ax.set_title("Multi-Agent Double Integrator with APF (Animation)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    
    def init():
        """初始化动画"""
        for line in lines:
            line.set_data([], [])
        for point in agents_points:
            point.set_data([], [])
        return lines + agents_points
    
    def animate(frame):
        """每一帧的更新函数"""
        for i in range(n_agents):
            # 如果轨迹长度足够，就更新
            if frame < len(trajectories[i]):
                # 绘制到当前帧的轨迹
                x_data = [trajectories[i][j][0] for j in range(frame + 1)]
                y_data = [trajectories[i][j][1] for j in range(frame + 1)]
                lines[i].set_data(x_data, y_data)
                
                # 更新智能体当前位置
                agents_points[i].set_data(trajectories[i][frame][0], trajectories[i][frame][1])
        
        return lines + agents_points
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=max_frames,
                         init_func=init, blit=True, interval=interval)
    
    # 保存动画(如果需要)
    if save_animation:
        anim.save('agents_animation.mp4', writer='ffmpeg', fps=30)
    
    plt.tight_layout()
    plt.show()
    
    return anim


# ==================================================
# 5) main 函数: 组织整体仿真流程
# ==================================================
def main():
    dt = 0.1
    max_steps = 500
    n_agents = 3

    alpha = 0.1   # 吸引势系数
    beta  = 7   # 排斥势系数
    Kp    = 1.0   # 位置误差增益
    Kv    = 1.0   # 速度衰减增益
    d0    = 5.0   # 安全距离(障碍影响范围)

    # ===== 圆形障碍物列表 =====
    obstacles = [
        (10.0,  5.0, 2.0),
        (15.0,  0.0, 2.0),
        (20.0,  4.0, 2.0),
    ]

    # ===== 多智能体初始状态 [x0, y0, vx0, vy0] =====
    init_states = np.array([
        [0.0,   0.0,   0.0,  0.0],  # Agent 1
        [0.0,   8.0,   0.0,  0.0],  # Agent 2
        [-2.0,  4.0,   0.0,  0.0],  # Agent 3
    ])

    # ===== 对应目标位置 [x_goal, y_goal] =====
    goals = np.array([
        [30.0,  5.0],   # Agent 1
        [28.0,  1.0],   # Agent 2
        [25.0,  8.0],   # Agent 3
    ])

    # 1) 创建环境
    env = Environment(obstacles, d0=d0)

    # 2) 创建控制器
    controller = APFControllerDoubleIntegrator(alpha=alpha, beta=beta, Kp=Kp, Kv=Kv, environment=env)

    # 3) 系统动力学(每个智能体一个)
    agents = [DoubleIntegratorSystem(init_states[i], dt=dt) for i in range(n_agents)]

    # 记录轨迹(仅记录位置)
    trajectories = [[] for _ in range(n_agents)]
    for i in range(n_agents):
        trajectories[i].append(agents[i].get_position().copy())

    # ===== 主循环 =====
    for step in range(max_steps):
        all_reached = True
        for i in range(n_agents):
            pos_i = agents[i].get_position()
            vel_i = agents[i].get_velocity()
            goal_i = goals[i]

            # 如果未到目标附近, 则计算控制
            if np.linalg.norm(pos_i - goal_i) > 0.5:
                all_reached = False
                u_i = controller.get_control(pos_i, vel_i, goal_i)
                agents[i].update(u_i)

            # 记录轨迹
            trajectories[i].append(agents[i].get_position().copy())

        if all_reached:
            print(f"All agents reached near their goals at step = {step}")
            break

    # ===== 可视化 =====
    # visualize(trajectories, obstacles, goals)  # 静态可视化
    
    # 使用动画可视化
    animate_agents(trajectories, obstacles, goals, interval=50)


# 若想直接运行此文件，请取消下面注释:
if __name__ == "__main__":
    main()

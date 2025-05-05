import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg

# ========== 1. Controller Classes ==========
class Controller:
    """Base controller class defining the interface"""
    def __init__(self):
        pass
    
    def compute_control(self, p_ref, p, v_ref, v):
        """Compute control input based on reference and current states"""
        raise NotImplementedError("Subclasses must implement compute_control")

class PDController(Controller):
    """PD controller implementation"""
    def __init__(self, Kp=1.5, Kd=1.0):
        super().__init__()
        self.Kp = Kp
        self.Kd = Kd
    
    def compute_control(self, p_ref, p, v_ref, v):
        p_error = p_ref - p
        v_error = v_ref - v
        return self.Kp * p_error + self.Kd * v_error

class LQRController(Controller):
    """LQR controller implementation"""
    def __init__(self, Q_pos=10.0, Q_vel=1.0, R=1.0):
        super().__init__()
        
        # State-space matrices for double integrator (position and velocity error)
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Cost matrices
        self.Q = np.diag([Q_pos, Q_pos, Q_pos, Q_vel, Q_vel, Q_vel])
        self.R = np.eye(3) * R
        
        # Compute LQR gain matrix K
        self.K = self._solve_lqr(self.A, self.B, self.Q, self.R)
    
    def _solve_lqr(self, A, B, Q, R):
        """Solve the continuous time LQR controller"""
        P = linalg.solve_continuous_are(A, B, Q, R)
        K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
        return K
    
    def compute_control(self, p_ref, p, v_ref, v):
        # Assemble state vector [e_x, e_y, e_z, edot_x, edot_y, edot_z]
        p_error = p_ref - p
        v_error = v_ref - v
        state = np.concatenate((p_error, v_error))
        
        # LQR control law: u = -K*x
        return np.dot(self.K, state)

# ========== 2. System Dynamics Class ==========
class UAVFormation:
    """Class handling UAV formation dynamics and simulation"""
    def __init__(self, n=2, dim=3, controller=None):
        self.n = n                  # Number of UAVs
        self.dim = dim              # Dimension of space (3D)
        self.controller = controller # Controller for follower UAVs
        
        # Formation reference offset - mainly distributed along X-axis
        self.delta_ref = np.array([10.0, 0.0, 0.0])
        
        # Leader UAV cruise velocity settings - mainly flying along X-axis
        self.cruise_velocity = np.array([5.0, 0.0, 0.0])
        
        # 3D sinusoidal disturbance settings
        self.dist_amplitude = np.array([0.2, 0.5, 0.3])
        self.dist_frequency = np.array([0.5, 0.3, 0.4])
        
        # Simulation time settings
        self.t_span = (0, 30)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 301)
        
        # Simulation results
        self.sol = None
        self.p_sol = None
        
    def dynamics(self, t, X):
        """UAV formation dynamics function"""
        # Parse positions and velocities
        p = np.zeros((self.n, self.dim))
        v = np.zeros((self.n, self.dim))
        for i in range(self.n):
            p[i, :] = X[i*self.dim:(i+1)*self.dim]
            v[i, :] = X[self.n*self.dim + i*self.dim : self.n*self.dim + (i+1)*self.dim]

        # Calculate control inputs
        u = np.zeros_like(v)
        for i in range(self.n):
            if i == 0:
                # Leader UAV - cruising with predefined velocity and 3D sinusoidal disturbance
                
                # Calculate target velocity (base cruise velocity + sinusoidal disturbance)
                target_v = self.cruise_velocity.copy()
                for j in range(self.dim):
                    target_v[j] += self.dist_amplitude[j] * np.sin(self.dist_frequency[j] * t)
                
                # Calculate target position (assuming starting from origin and integrating target velocity)
                target_p = np.zeros(self.dim)
                for j in range(self.dim):
                    # Integrate base velocity to get position
                    target_p[j] = self.cruise_velocity[j] * t
                    # Integrate sinusoidal disturbance to get position offset
                    if self.dist_frequency[j] > 0:
                        target_p[j] += (self.dist_amplitude[j] / self.dist_frequency[j]) * (1 - np.cos(self.dist_frequency[j] * t))
                
                # Position and velocity errors
                p_error = target_p - p[i]
                v_error = target_v - v[i]
                
                # PD control for leader
                u[i] = 2.0 * p_error + 1.0 * v_error
            else:
                # Follower UAVs - maintain relative position to preceding UAV using selected controller
                p_ref = p[i-1] - self.delta_ref  # Reference position (relative to preceding UAV)
                v_ref = v[i-1]                   # Reference velocity (match preceding UAV)
                u[i] = self.controller.compute_control(p_ref, p[i], v_ref, v[i])

        # Update derivatives
        dX = np.zeros_like(X)
        for i in range(self.n):
            dX[i*self.dim:(i+1)*self.dim] = v[i]
            dX[self.n*self.dim + i*self.dim : self.n*self.dim + (i+1)*self.dim] = u[i]

        return dX
    
    def initialize_states(self):
        """Initialize UAV positions and velocities"""
        # Initial positions - set to equilibrium state
        p0 = np.zeros((self.n, self.dim))
        # Set leader UAV position (can be any position, here choosing origin)
        p0[0, :] = np.zeros(self.dim)
        # Set follower UAV equilibrium positions - each UAV behind the previous one
        for i in range(1, self.n):
            p0[i, :] = p0[i-1, :] - self.delta_ref  # Using addition instead of subtraction

        # Initial velocities - all UAVs start with the same cruise velocity
        v0 = np.zeros((self.n, self.dim))
        for i in range(self.n):
            v0[i, :] = self.cruise_velocity.copy()

        # Combine state vector
        return np.hstack([p0.flatten(), v0.flatten()])
    
    def run_simulation(self):
        """Run the UAV formation simulation"""
        X0 = self.initialize_states()
        self.sol = solve_ivp(
            fun=self.dynamics, 
            t_span=self.t_span, 
            y0=X0, 
            t_eval=self.t_eval, 
            method='RK45', 
            rtol=1e-6
        )
        self.p_sol = self.sol.y[0:self.n*self.dim, :]
        return self.sol

# ========== 3. Visualization Class ==========
class FormationVisualizer:
    """Class for visualizing UAV formation flight"""
    def __init__(self, formation):
        self.formation = formation
        self.n = formation.n
        self.dim = formation.dim
        self.p_sol = formation.p_sol
        self.sol = formation.sol
        self.colors = plt.cm.jet(np.linspace(0, 1, self.n))
        
        # Compute boundaries for plots
        self.max_x = np.max(self.p_sol[0:self.n*self.dim:self.dim, :]) + 10
        self.min_x = np.min(self.p_sol[0:self.n*self.dim:self.dim, :]) - 10
        self.max_y = np.max(self.p_sol[1:self.n*self.dim:self.dim, :]) + 10
        self.min_y = np.min(self.p_sol[1:self.n*self.dim:self.dim, :]) - 10
        self.max_z = np.max(self.p_sol[2:self.n*self.dim:self.dim, :]) + 5
        self.min_z = np.min(self.p_sol[2:self.n*self.dim:self.dim, :]) - 1
        
    def plot_complete_trajectory(self, controller_name="PD"):
        """Plot complete trajectory for all UAVs"""
        fig_traj = plt.figure(figsize=(12, 10))
        ax_traj = fig_traj.add_subplot(111, projection='3d')
        
        # Plot complete trajectory for each UAV
        for i in range(self.n):
            # Extract current UAV trajectory
            x_traj = self.p_sol[i*self.dim, :]
            y_traj = self.p_sol[i*self.dim+1, :]
            z_traj = self.p_sol[i*self.dim+2, :]
            
            # Plot complete trajectory
            ax_traj.plot(x_traj, y_traj, z_traj, '-', linewidth=2, color=self.colors[i], 
                         label=f'UAV {i} Trajectory')
            
            # Mark start and end points
            ax_traj.scatter(x_traj[0], y_traj[0], z_traj[0], s=80, color=self.colors[i], marker='o', edgecolors='black')
            ax_traj.scatter(x_traj[-1], y_traj[-1], z_traj[-1], s=80, color=self.colors[i], marker='s', edgecolors='black')
            
            # Add annotations
            ax_traj.text(x_traj[0], y_traj[0], z_traj[0], f'Start {i}', fontsize=10)
            ax_traj.text(x_traj[-1], y_traj[-1], z_traj[-1], f'End {i}', fontsize=10)

        # Plot formation structure at regular time intervals (every 5 seconds)
        interval_indices = np.linspace(0, len(self.sol.t)-1, 7).astype(int)
        for idx in interval_indices:
            t_val = self.sol.t[idx]
            for i in range(self.n):
                x = self.p_sol[i*self.dim, idx]
                y = self.p_sol[i*self.dim+1, idx]
                z = self.p_sol[i*self.dim+2, idx]
                
                # Small scatter points to mark this time point
                ax_traj.scatter(x, y, z, s=30, color='black', alpha=0.7)
                
                # If leader UAV, add time annotation
                if i == 0:
                    ax_traj.text(x, y, z+0.5, f't={t_val:.1f}s', fontsize=8)
                
                # Add connecting lines at certain time points to show formation structure
                if i < self.n-1:
                    next_x = self.p_sol[(i+1)*self.dim, idx]
                    next_y = self.p_sol[(i+1)*self.dim+1, idx]
                    next_z = self.p_sol[(i+1)*self.dim+2, idx]
                    ax_traj.plot([x, next_x], [y, next_y], [z, next_z], 'k--', linewidth=0.5, alpha=0.5)
        
        # Set figure boundaries and labels
        ax_traj.set_xlim([self.min_x, self.max_x])
        ax_traj.set_ylim([self.min_y, self.max_y])
        ax_traj.set_zlim([self.min_z, self.max_z])
        
        ax_traj.set_xlabel('X [m]')
        ax_traj.set_ylabel('Y [m]')
        ax_traj.set_zlabel('Z [m]')
        ax_traj.set_title(f'X-Axis Formation Flight Complete Trajectory ({controller_name} Controller)')
        
        ax_traj.legend(loc='upper right')
        ax_traj.grid(True)
        
        plt.savefig(f'output_figures/formation_flight_trajectory_{controller_name}.png', dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        
    def animate_formation(self, controller_name="PD"):
        """Create dynamic animation of UAV formation flight"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set figure boundaries - adjust to fit the cruise trajectory
        ax.set_xlim([self.min_x, self.max_x])
        ax.set_ylim([self.min_y, self.max_y])
        ax.set_zlim([self.min_z, self.max_z])
        
        # Initialize UAV position scatter points and trajectory lines
        scatters = []
        trails = []
        trail_length = 50  # Trajectory length
        
        for i in range(self.n):
            # Scatter points represent UAVs
            scatter = ax.scatter([], [], [], s=100, color=self.colors[i], 
                                label=f'UAV {i}', edgecolors='black')
            scatters.append(scatter)
            
            # Lines represent trajectories
            trail, = ax.plot([], [], [], color=self.colors[i], alpha=0.5, linewidth=1)
            trails.append(trail)
        
        # Add axis labels and title
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'X-Axis Formation Flight Dynamic Simulation ({controller_name} Controller)')
        
        # Add legend and grid
        ax.legend(loc='upper right')
        ax.grid(True)
        
        def update(frame):
            for i in range(self.n):
                # Update UAV positions
                x = self.p_sol[i*self.dim, frame]
                y = self.p_sol[i*self.dim+1, frame]
                z = self.p_sol[i*self.dim+2, frame]
                
                scatters[i]._offsets3d = ([x], [y], [z])
                
                # Update trajectories
                start_idx = max(0, frame - trail_length)
                x_trail = self.p_sol[i*self.dim, start_idx:frame+1]
                y_trail = self.p_sol[i*self.dim+1, start_idx:frame+1]
                z_trail = self.p_sol[i*self.dim+2, start_idx:frame+1]
                
                trails[i].set_data(x_trail, y_trail)
                trails[i].set_3d_properties(z_trail)
            
            # Update title to display current time
            ax.set_title(f'X-Axis Formation Flight ({controller_name}) - Time: {self.sol.t[frame]:.1f}s')
            
            return scatters + trails
        
        # Create animation
        ani = FuncAnimation(
            fig, update, frames=len(self.sol.t), 
            interval=50, blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return ani  # Return animation object to prevent garbage collection

# ========== 4. Main Function ==========
def main():
    # Select controller type
    controller_type = "LQR"  # Options: "PD", "LQR"
    
    # Create controller instance
    if controller_type == "PD":
        controller = PDController(Kp=1.5, Kd=1.0)
    elif controller_type == "LQR":
        controller = LQRController(Q_pos=10.0, Q_vel=1.0, R=1.0)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Create UAV formation instance and run simulation
    formation = UAVFormation(n=5, dim=3, controller=controller)
    formation.run_simulation()
    
    # Create visualizer and generate visualizations
    visualizer = FormationVisualizer(formation)
    visualizer.plot_complete_trajectory(controller_name=controller_type)
    ani = visualizer.animate_formation(controller_name=controller_type)
    
    # Uncomment to save animation
    # ani.save(f'3d_formation_flight_{controller_type}.mp4', writer='ffmpeg', fps=20, dpi=200)
    
    return formation, visualizer, ani

# ========== 5. Run the program ==========
if __name__ == "__main__":
    formation, visualizer, ani = main()

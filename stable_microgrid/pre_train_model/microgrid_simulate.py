import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class MicrogridSimulator:
    def __init__(
        self,
        n: int,
        V: List[float],
        B: List[List[float]],
        P_L: List[float],
        P_star: List[float],
        omega_star: float,
        tau: List[float],
        eta: List[float],
        k: List[float],
        G: List[List[float]],
        L: List[List[float]]
    ):
        """
        Initialize the microgrid simulator.
        
        Args:
            n: Number of inverters
            V: List of voltage magnitudes
            B: Matrix of line susceptances
            P_L: List of local load demands
            P_star: List of desired power injections
            omega_star: Desired frequency
            tau: List of time constants
            eta: List of droop gains
            k: List of secondary control gains
            G: Communication graph adjacency matrix
            L: Voltage angle communication graph adjacency matrix
        """
        self.n = n
        self.V = np.array(V)
        self.B = np.array(B)
        self.P_L = np.array(P_L)
        self.P_star = np.array(P_star)
        self.omega_star = omega_star
        self.tau = np.array(tau)
        self.eta = np.array(eta)
        self.k = np.array(k)
        self.G = np.array(G)
        self.L = np.array(L)
        
        # Initialize state variables
        self.delta = np.zeros(n)  # Voltage phase angles
        self.omega = np.zeros(n)  # Frequencies
        self.xi = np.zeros(n)     # Secondary controller states
        
    def compute_power_flow(self, delta: np.ndarray) -> np.ndarray:
        """Compute power flow at each node."""
        P_I = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.B[i,j] != 0:
                    P_I[i] += abs(self.B[i,j]) * self.V[i] * self.V[j] * np.sin(delta[i] - delta[j])
        return self.P_L + P_I
    
    def update_loads(self, t: float):
        """Update loads based on time."""
        base_load = 1260.0  # Base load in Watts
        
        if 0 <= t < 10:
            # Increased loads during 0 <= t < 1500
            self.P_L[0] = base_load * 1.20  # 20% increase for inverter 1
            for i in range(1, self.n, 2):  # Even numbered inverters (2,4,...)
                self.P_L[i] = base_load * 1.10  # 10% increase
            for i in range(2, self.n, 2):  # Odd numbered inverters (3,5,...)
                self.P_L[i] = base_load
        else:
            # Return to rated values at t >= 1500
            self.P_L = np.full(self.n, base_load)

    def secondary_droop_control(self, omega: np.ndarray, xi: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Compute the secondary droop control input.
        
        Args:
            omega: Current frequencies
            xi: Current secondary controller states
            delta: Current phase angles
            
        Returns:
            Control input for each inverter
        """
        omega_diff = omega - self.omega_star
        xi_diff = np.zeros(self.n)
        delta_diff = np.zeros(self.n)
        
        # Compute the differences for communication terms
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    xi_diff[i] += self.G[i,j] * (xi[i] - xi[j])
                    delta_diff[i] += self.L[i,j] * (delta[i] - delta[j])
        
        # Compute control input
        u = (1/self.k) * (
            -omega_diff
            - xi_diff
            - delta_diff
        )
        
        return u

    def step(self, dt: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate one step of the microgrid dynamics.
        
        Args:
            dt: Time step size
            t: Current simulation time
            
        Returns:
            Tuple of (delta, omega, xi) after the step
        """
        # Update loads based on current time
        self.update_loads(t)
        
        # Compute power flow
        P = self.compute_power_flow(self.delta)
        
        # Update states using Euler integration
        # Phase angle dynamics
        self.delta += self.omega * dt
        
        # Frequency dynamics
        self.omega += (
            (1/self.tau) * (
                -(self.omega - self.omega_star)
                - self.eta * (P - self.P_star)
                + self.xi
            )
        ) * dt
        
        # Secondary controller dynamics
        u = self.secondary_droop_control(self.omega, self.xi, self.delta)
        self.xi += u * dt
        
        return self.delta.copy(), self.omega.copy(), self.xi.copy()
    
    def simulate(self, t_end: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the microgrid for a given time period.
        
        Args:
            t_end: End time of simulation
            dt: Time step size
            
        Returns:
            Tuple of (time, delta, omega, xi) arrays
        """
        t = np.arange(0, t_end + dt, dt)
        delta_history = np.zeros((len(t), self.n))
        omega_history = np.zeros((len(t), self.n))
        xi_history = np.zeros((len(t), self.n))
        
        for i in range(len(t)):
            delta_history[i], omega_history[i], xi_history[i] = self.step(dt, t[i])
            
        return t, delta_history, omega_history, xi_history 

    def step_without_update_loads(self, dt: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate one step of the microgrid dynamics without updating loads.
        """
        # Compute power flow
        P = self.compute_power_flow(self.delta)
        
        # Update states using Euler integration
        # Phase angle dynamics
        self.delta += self.omega * dt
        
        # Frequency dynamics
        self.omega += (
            (1/self.tau) * (
                -(self.omega - self.omega_star)
                - self.eta * (P - self.P_star)
                + self.xi
            )
        ) * dt
        
        # Secondary controller dynamics
        u = self.secondary_droop_control(self.omega, self.xi, self.delta)
        self.xi += u * dt
        
        return self.delta.copy(), self.omega.copy(), self.xi.copy()
        

    def plot_states(self, t: np.ndarray, delta: np.ndarray, omega: np.ndarray, xi: np.ndarray):
        """
        Plot the system states over time.
        
        Args:
            t: Time array
            delta: Phase angle history
            omega: Frequency history
            xi: Secondary controller state history
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot phase angles
        for i in range(self.n):
            ax1.plot(t, delta[:, i], label=f'Inverter {i+1}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Phase Angle (rad)')
        ax1.set_title('Phase Angles')
        ax1.grid(True)
        ax1.legend()
        
        # Plot frequencies
        for i in range(self.n):
            ax2.plot(t, omega[:, i], label=f'Inverter {i+1}')
        ax2.axhline(y=self.omega_star, color='k', linestyle='--', label='ω*')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (rad/s)')
        ax2.set_title('Frequencies')
        ax2.grid(True)
        ax2.legend()
        
        # Plot secondary controller states
        for i in range(self.n):
            ax3.plot(t, xi[:, i], label=f'Inverter {i+1}')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Secondary Controller State')
        ax3.set_title('Secondary Controller States')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        plt.savefig('output_figures/microgrid_states.png', dpi=300)

if __name__ == "__main__":
    # System parameters from Table II
    n = 3  # Number of inverters
    V = [325.3] * n  # Voltage magnitude (V)
    B = [[-0.0056, -0.0112, 0],
         [-0.0112, -0.0151, -0.0039],
         [0, -0.0039, -0.0112]]  # Line susceptances
    P_L = [1260.0] * n  # Initial loads (W)
    P_star = [1260.0] * n  # Desired power injections (W)
    omega_star = 2 * np.pi * 50  # Desired frequency (rad/s) - 50 Hz

    # Controller parameters from Table I
    # Using DSS Controller C1 parameters
    tau = [1.4895] * n  # Time constants
    eta = [6.3509e-4] * n  # Droop gains
    k = [4.9481] * n  # Secondary control gains
    
    # Communication graph parameters
    g_val = 0.0213  # g_{i,i±1} value from Table I
    l_val = 0.0043  # l_{i,i±1} value from Table I
    
    # Construct communication graphs
    G = np.zeros((n, n))
    L = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            G[i,i-1] = G[i-1,i] = -g_val
            L[i,i-1] = L[i-1,i] = -l_val
        if i < n-1:
            G[i,i+1] = G[i+1,i] = -g_val
            L[i,i+1] = L[i+1,i] = -l_val
        G[i,i] = -np.sum(G[i,:])  # Make row sum zero
        L[i,i] = -np.sum(L[i,:])  # Make row sum zero

    # Initialize simulator
    simulator = MicrogridSimulator(n, V, B, P_L, P_star, omega_star, tau, eta, k, G, L)

    # Set initial conditions from Table II
    simulator.delta = np.zeros(n)  # δᵢ(0) = 0
    simulator.omega = np.array([2 * np.pi * 50] * n)  # ωᵢ(0) = 50 Hz
    simulator.xi = np.zeros(n)  # ξᵢ(0) = 0

    # Simulate for 2000 seconds to observe load change and recovery
    t, delta, omega, xi = simulator.simulate(t_end=100.0, dt=0.01)
    
    # Plot the results
    simulator.plot_states(t, delta, omega, xi)
    

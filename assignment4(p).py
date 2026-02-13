import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu

class HeatSlabSolver:
    def __init__(self, L=1.0, nx=51, alpha=1.0, t_init=350.0, t_left=300.0, t_right=400.0):
        """
        Initialize the 1D Heat Slab problem parameters.
        """
        # Physics & Geometry
        self.L = L
        self.alpha = alpha
        self.T_init = float(t_init)  # Ensure float
        self.T_left = float(t_left)
        self.T_right = float(t_right)
        
        # Discretization
        self.nx = nx
        self.x = np.linspace(0, L, nx)
        self.dx = self.x[1] - self.x[0]
        
        # Internal node count (for linear algebra systems)
        self.n_inner = nx - 2

    def analytical_solution(self, t, n_terms=100):
        """Computes the exact Fourier series solution at time t."""
        if t <= 0: return np.full_like(self.x, self.T_init)
        
        # 1. Steady State Solution (Linear gradient)
        T_steady = self.T_left + (self.T_right - self.T_left) * (self.x / self.L)
        
        # 2. Transient Deviation (Fourier Series)
        summation = np.zeros_like(self.x)
        for n in range(1, n_terms + 1):
            lambda_n = n * np.pi / self.L
            
            # Integral of Initial Condition deviation
            term_init = (self.T_init / lambda_n) * (1 - (-1)**n)
            term_bc   = (self.T_left - (-1)**n * self.T_right) / lambda_n
            Bn = term_init - term_bc 
            
            summation += Bn * np.sin(lambda_n * self.x) * np.exp(-self.alpha * lambda_n**2 * t)
            
        return T_steady - (2/self.L) * summation

    def _build_matrices(self, r, scheme):
        """Helper to construct sparse matrices for time-stepping."""
        I = sparse.eye(self.n_inner, format='csc')
        # Second derivative operator (central difference)
        diagonals = [np.ones(self.n_inner-1), -2*np.ones(self.n_inner), np.ones(self.n_inner-1)]
        D2 = sparse.diags(diagonals, [-1, 0, 1], format='csc')
        
        if scheme == 'implicit':
            # (I - r*D2) * T_new = T_old
            A = I - r * D2
            return A, None
        
        elif scheme == 'cn':
            # (I - 0.5*r*D2) * T_new = (I + 0.5*r*D2) * T_old
            A = I - 0.5 * r * D2
            B = I + 0.5 * r * D2
            return A, B
        return None, None

    def solve(self, scheme, dt, target_times):
        """
        Main solver method.
        scheme: 'explicit', 'implicit', 'cn'
        """
        # --- FIX IS HERE: dtype=np.float64 ---
        T = np.full(self.nx, self.T_init, dtype=np.float64) 
        T[0], T[-1] = self.T_left, self.T_right
        
        r = self.alpha * dt / self.dx**2
        
        # Slightly increase steps to ensure we cover the last target time
        nt = int(max(target_times) / dt) + 10
        
        # Prepare storage
        results = {}
        # Map target times to specific step numbers
        target_indices = [int(t/dt) for t in target_times]
        
        # Pre-compute matrices for Implicit/CN
        solve_A = None
        B = None
        if scheme in ['implicit', 'cn']:
            A_mat, B_mat = self._build_matrices(r, scheme)
            solve_A = splu(A_mat).solve  # Pre-factorize for speed
            B = B_mat
        
        # Time Stepping Loop
        for n in range(1, nt + 1):
            
            if scheme == 'explicit':
                # T[i] = T[i] + r*(T[i+1] - 2T[i] + T[i-1])
                # Using temp array to avoid overwriting while reading
                T_new_inner = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
                T[1:-1] = T_new_inner
                
            elif scheme == 'implicit':
                # RHS is T_old + boundary contributions
                b_vec = T[1:-1].copy()
                b_vec[0]  += r * self.T_left
                b_vec[-1] += r * self.T_right
                T[1:-1] = solve_A(b_vec)
                
            elif scheme == 'cn':
                # RHS is B*T_old + boundary contributions
                b_vec = B.dot(T[1:-1])
                b_vec[0]  += r * self.T_left
                b_vec[-1] += r * self.T_right
                T[1:-1] = solve_A(b_vec)
            
            # Store result if this step corresponds to a target time
            # We check closeness to handle floating point drift
            current_time = n * dt
            for t_targ in target_times:
                if abs(current_time - t_targ) < dt/1.5:
                    results[t_targ] = T.copy()

        return results

# --- Comparison & Plotting ---

def compare_schemes():
    # Configuration
    L = 1.0
    alphas = [0.0001, 0.001, 0.01]
    times = [1, 5, 10, 50, 100]
    
    # Create subplots
    fig, axes = plt.subplots(len(alphas), 1, figsize=(10, 15), constrained_layout=True)
    if len(alphas) == 1: axes = [axes]
    
    for ax, alpha in zip(axes, alphas):
        print(f"Simulating Alpha = {alpha}...")
        
        # Instantiate Solver
        solver = HeatSlabSolver(L=L, alpha=alpha, nx=51)
        
        # 1. Analytical Solution
        ana_res = {t: solver.analytical_solution(t) for t in times}
        
        # 2. Numerical Solutions
        # Explicit Stability Limit: dt <= 0.5 * dx^2 / alpha
        dt_stable = 0.5 * solver.dx**2 / alpha
        
        # Run solvers
        # Note: We use 90% of stable dt for explicit. 
        # Implicit/CN are unconditionally stable, but we use the same dt for comparison.
        res_exp = solver.solve('explicit', dt=dt_stable * 0.9, target_times=times)
        res_imp = solver.solve('implicit', dt=dt_stable * 0.9, target_times=times)
        res_cn  = solver.solve('cn',       dt=dt_stable * 0.9, target_times=times)
        
        # Plotting
        colors = plt.cm.jet(np.linspace(0, 0.9, len(times)))
        
        for i, t in enumerate(times):
            if t not in res_exp: continue # Skip if time not reached
            
            c = colors[i]
            # Analytical (Line)
            ax.plot(solver.x, ana_res[t], color=c, linestyle='-', label=f't={t}s' if i==0 else "")
            
            # Explicit (Dots)
            ax.plot(solver.x[::4], res_exp[t][::4], 'o', color=c, fillstyle='none', markersize=6)
            
            # Implicit (Triangles)
            ax.plot(solver.x[1::4], res_imp[t][1::4], '^', color=c, fillstyle='none', markersize=6)
            
            # CN (Crosses)
            ax.plot(solver.x[2::4], res_cn[t][2::4], 'x', color=c, markersize=6)

        ax.set_title(f"Alpha = {alpha} $m^2/s$")
        ax.set_ylabel("Temperature (K)")
        ax.grid(True, alpha=0.3)
        if alpha == alphas[-1]: ax.set_xlabel("Position (m)")
    
    # Global Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='k', lw=2, label='Analytical'),
        Line2D([0], [0], color='k', marker='o', lw=0, fillstyle='none', label='Explicit'),
        Line2D([0], [0], color='k', marker='^', lw=0, fillstyle='none', label='Implicit'),
        Line2D([0], [0], color='k', marker='x', lw=0, label='Crank-Nicolson')
    ]
    fig.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(0.98, 0.99))
    plt.show()

if __name__ == "__main__":
    compare_schemes()
import jax
import jax.numpy as jnp 
from jax import lax  # For parallelization with JAX
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from functools import partial
from mpl_toolkits.mplot3d import Axes3D


"""TARGET Moments and Computed Moments have a Severe Descreprency"""




# Potential V and dV
def potential_V(x, y):
    # V(x,y) = x^4 + y^4 + x^3 - 2 x y^2 + a(x^2 + y^2) + theta1 x + theta2 y
    a      = 2.0 
    theta1 = 5.0
    theta2 = -5.0
    return x**4 + y**4 + x**3 - 2.0*x*y**2 + a*(x**2 + y**2) + theta1*x + theta2*y

def partial_deriv_of_V(x, y):
    # dV/dx = 4 x^3 + 3 x^2 - 2 y^2 + 2 a x + theta1
    # dV/dy = 4 y^3 - 4 x y + 2 a y + theta2
    a      = 2.0 
    theta1 = 5.0
    theta2 = -5.0
    dVdx = 4.0*x**3 + 3.0*x**2 - 2.0*y**2 + 2.0*a*x + theta1
    dVdy = 4.0*y**3 - 4.0*x*y + 2.0*a*y + theta2
    return dVdx, dVdy

def control(t, params=None):
    return 0.0, 0.0




# Maximum Entropy 
@jax.jit
def p_lam_unnorm(x, y, lam):
    # lam is the vector of Lagrange multipliers
    (l1, l2, l11, l22, l12, l3x, l3y, l21) = lam
    exponent = (l1 * x + l2 * y +
                l11 * (x**2) + l22 * (y**2) + l12 * (x * y) +
                l3x * (x**3) + l3y * (y**3) +
                l21 * (x**2 * y))
    e_clamped = jnp.clip(-exponent, -50.0, 50.0)
    return jnp.exp(e_clamped)

@jax.jit
def compute_single_moment(lam, n, m, xgrid, ygrid):
    XX, YY = jnp.meshgrid(xgrid, ygrid, indexing='xy')
    pdf_unn = p_lam_unnorm(XX, YY, lam)
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    # Computing the normalization constant Z 
    Z  = jnp.sum(pdf_unn) * dx * dy + 1e-16
    # Computint the moment E[x^n y^m]
    val = (jnp.sum((XX**n) * (YY**m) * pdf_unn) * dx * dy) / Z
    return val

def compute_moments_multicalls(lam, poly_orders, xgrid, ygrid):
    # Loops over every (n, m) pair to compute moments 
    results = []
    for (n, m) in poly_orders:
        val = compute_single_moment(lam, n, m, xgrid, ygrid)
        results.append(val)
    return jnp.array(results)

def residual_function(lam, target_mom, poly_orders, xgrid, ygrid):
    # Computes moments of current lambda and subtract the target moments
    comp = compute_moments_multicalls(lam, poly_orders, xgrid, ygrid)
    return comp - target_mom

@partial(jax.jit, static_argnames=('max_steps', 'step_size', 'tol'))
def run_lagrange_minimization(lam_init, target_mom, poly_orders, xgrid, ygrid, max_steps=40, step_size=1e-4, tol=1e-7):
    def cond_fun(value):
        lam_curr, step_idx, rnorm = value
        return (step_idx < max_steps) & (rnorm > tol)
    def body_fun(value):
        lam_curr, step_idx, _ = value
        R = residual_function(lam_curr, target_mom, poly_orders, xgrid, ygrid)
        rnow = jnp.sqrt(jnp.sum(R**2))
        def loss_fn(lc):
            Rc = residual_function(lc, target_mom, poly_orders, xgrid, ygrid)
            return 0.5 * jnp.sum(Rc**2)
        grad_l = jax.grad(loss_fn)(lam_curr)
        lam_new = lam_curr - step_size * grad_l
        return (lam_new, step_idx + 1, rnow)
    R0 = residual_function(lam_init, target_mom, poly_orders, xgrid, ygrid)
    r0 = jnp.sqrt(jnp.sum(R0**2))
    init_val = (lam_init, 0, r0)
    lam_final, _, _ = lax.while_loop(cond_fun, body_fun, init_val)
    return lam_final

def run_lagrange_minimization_debug(lam_init, target_mom, poly_orders, xgrid, ygrid, max_steps=40, step_size=1e-4, tol=1e-7):
    # Track residuals
    lam_curr = lam_init
    residuals = []
    for step_idx in range(max_steps):
        R = residual_function(lam_curr, target_mom, poly_orders, xgrid, ygrid)
        rnow = jnp.sqrt(jnp.sum(R**2)).item()
        residuals.append(rnow)
        if rnow < tol:
            break
        def loss_fn(lc):
            Rc = residual_function(lc, target_mom, poly_orders, xgrid, ygrid)
            return 0.5 * jnp.sum(Rc**2)
        grad_l = jax.grad(loss_fn)(lam_curr)
        lam_curr = lam_curr - step_size * grad_l
    return lam_curr, np.array(residuals)

def maxent_rhs_8moments(M, t, D, xgrid, ygrid, control_fn, control_params):
    # M = [E[x], E[y], E[x^2], E[y^2], E[xy], E[x^3], E[y^3], E[x^2 y]]
    (Ex, Ey, Ex2, Ey2, Exy, Ex3, Ey3, Ex2y) = M
    # Polynomial orders corresponding to the moments in our MaxEnt PDF
    poly_orders = [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (3, 0), (0, 3), (2, 1)]
    # Initializing the target moments from the current moment vector
    target_mom = jnp.array([Ex, Ey, Ex2, Ey2, Exy, Ex3, Ey3, Ex2y], dtype=jnp.float32)
    # We need to initialize our lambda vectors with 0
    lam_init = jnp.zeros(8, dtype=jnp.float32)
    # Solving for the Lagrange multipliers that yield our target moments
    lam_final = run_lagrange_minimization(lam_init, target_mom, poly_orders, xgrid, ygrid, max_steps=40, step_size=1e-4, tol=1e-7)
    a = 2.0
    theta1 = 5.0
    theta2 = -5.0
    # Get control inputs at time t
    U1, U2 = control_fn(t, control_params)
  
    # Moments for dE[x]/dt 
    needed = [(3, 0), (2, 0), (0, 2), (1, 0)]
    cvals = compute_moments_multicalls(lam_final, needed, xgrid, ygrid)
    x3_, x2_, y2_, x_ = cvals
    # Using derived moments to compute E[dV/dx]
    EdVdx = 4.0 * x3_ + 3.0 * x2_ - 2.0 * y2_ + 2.0 * a * x_ + theta1
    dEx_dt = -EdVdx + U1
  
    # Moments from dE[y]/dt using E[dV/dy]
    needed2 = [(0, 3), (1, 1), (0, 1)]
    cvals2 = compute_moments_multicalls(lam_final, needed2, xgrid, ygrid)
    y3_, xy_, y_2 = cvals2
    EdVdy = 4.0 * y3_ - 4.0 * xy_ + 2.0 * a * y_2 + theta2
    dEy_dt = -EdVdy + U2
  
    # Moments for dE[x^2]/dt using x*dV/dx
    needed3 = [(4, 0), (3, 0), (1, 2), (2, 0), (1, 0)] 
    cvals3 = compute_moments_multicalls(lam_final, needed3, xgrid, ygrid)
    x4_, x3_2, x1y2_, x2_2, x_2 = cvals3
    ExdVdx = 4.0 * x4_ + 3.0 * x3_2 - 2.0 * x1y2_ + 2.0 * a * x2_2 + theta1 * x_2
    dEx2_dt = -2.0 * ExdVdx + 2.0 * U1 * Ex + 2.0 * D
  
    # Moments from dE[y^2]/dt using y*dV/dy
    needed4 = [(0, 4), (1, 2), (0, 2), (0, 1)]
    cvals4 = compute_moments_multicalls(lam_final, needed4, xgrid, ygrid)
    y4_, x1y2_2, y2_2, y_3 = cvals4
    EydVdy = 4.0 * y4_ - 4.0 * x1y2_2 + 2.0 * a * y2_2 + theta2 * y_3
    dEy2_dt = -2.0 * EydVdy + 2.0 * U2 * Ey + 2.0 * D
  
    # Moments from dE[xy]/dt from combined x*dV/dy and y*dV/dx terms
    need5a = [(1, 3), (2, 1), (1, 1)]
    vals5a = compute_moments_multicalls(lam_final, need5a, xgrid, ygrid)
    x1y3_, x2y1_, xy_2 = vals5a
    ex_xy0 = compute_single_moment(lam_final, 1, 0, xgrid, ygrid)
    ExdVdy = 4.0 * x1y3_ - 4.0 * x2y1_ + 2.0 * a * xy_2 + theta2 * ex_xy0
    need5b = [(3, 1), (2, 1), (1, 2), (1, 0), (0, 1)]
    vals5b = compute_moments_multicalls(lam_final, need5b, xgrid, ygrid)
    x3y1_, x2y1_2, x1y2_3, x1_, y1_ = vals5b
    ex_xy1 = compute_single_moment(lam_final, 1, 1, xgrid, ygrid)
    Ey_dVdx = 4.0 * x3y1_ + 3.0 * x2y1_2 - 2.0 * x1y2_3 + 2.0 * a * ex_xy1 + theta1 * y1_
    dExy_dt = -(ExdVdy + Ey_dVdx) + U1 * Ey + U2 * Ex + 2.0 * D
  
    # dE[x^3]/dt from x^2*dV/dx terms
    need6 = [(5, 0), (4, 0), (3, 2), (3, 0), (2, 0)]
    vals6 = compute_moments_multicalls(lam_final, need6, xgrid, ygrid)
    x5_, x4_, x3y2_, x3_3, x2_3 = vals6
    x2dVdx = 4.0 * x5_ + 3.0 * x4_ - 2.0 * x3y2_ + 2.0 * a * x3_3 + theta1 * x2_3
    dEx3_dt = -3.0 * x2dVdx + 3.0 * U1 * Ex2 + 6.0 * D
  
    # dE[y^3]/dt from y^2*dV/dy terms
    need7 = [(0, 5), (1, 3), (0, 3), (0, 2)]
    vals7 = compute_moments_multicalls(lam_final, need7, xgrid, ygrid)
    y5_, x1y3_2, y3_2, y2_3 = vals7
    y2dVdy = 4.0 * y5_ - 4.0 * x1y3_2 + 2.0 * a * y3_2 + theta2 * y2_3
    dEy3_dt = -3.0 * y2dVdy + 3.0 * U2 * Ey2 + 6.0 * D
  
    # dE[x^2 y]/dt from mixed terms
    need8a = [(4, 1), (3, 1), (1, 3), (2, 1), (1, 1)]
    vals8a = compute_moments_multicalls(lam_final, need8a, xgrid, ygrid)
    x4y1_, x3y1_, x1y3_, x2y1_, xy_1 = vals8a
    xydVdx = 4.0 * x4y1_ + 3.0 * x3y1_ - 2.0 * x1y3_ + 2.0 * a * x2y1_ + theta1 * xy_1
    need8b = [(2, 3), (3, 1), (2, 1), (2, 0), (0, 1)]
    vals8b = compute_moments_multicalls(lam_final, need8b, xgrid, ygrid)
    x2y3_, x3y1_2, x2y1_2, x2_4, y1_2 = vals8b
    x2dVdy_2 = 4.0 * x2y3_ - 4.0 * x3y1_2 + 2.0 * a * x2y1_2 + theta2 * (compute_single_moment(lam_final, 2, 1, xgrid, ygrid))
    dEx2y_dt = -2.0 * xydVdx - x2dVdy_2 + U1 * Exy + U2 * Ex2 + 4.0 * D
    return jnp.array([dEx_dt, dEy_dt, dEx2_dt, dEy2_dt, dExy_dt, dEx3_dt, dEy3_dt, dEx2y_dt], dtype=jnp.float32), lam_final

def solve_maxent_closure_3rd_order(D=0.2, dt=1e-6, T=1e-4, Nx=40, Ny=40, L=2.0, control_fn=None, control_params=None, M0=None, debug_interval=10):
    if control_fn is None:
        control_fn = lambda t, p: (0.0, 0.0)
    xgrid = jnp.linspace(-L, L, Nx)
    ygrid = jnp.linspace(-L, L, Ny)
    # Changed initial moment vector: second-order moments set to 1.0 for realistic variance
    if M0 is None:
        M0 = jnp.array([0., 0., 1.0, 1.0, 0., 0., 0., 0.], dtype=jnp.float32)
    n_steps = int(T / dt)
    t_array = jnp.linspace(0, T, n_steps + 1)
    def body_fn(M, step_idx):
        t_current = dt * step_idx
        dMdt, lam_val = maxent_rhs_8moments(M, t_current, D, xgrid, ygrid, control_fn, control_params)
        M_next = M + dt * dMdt
        M_next = jnp.where(jnp.isnan(M_next), M, M_next)
        return M_next, (M_next, lam_val)
    final_carry, outputs = lax.scan(body_fn, M0, jnp.arange(n_steps))
    M_out, lam_out = outputs  # M_out has shape (n_steps, 8), lam_out has shape (n_steps, 8)
    M_traj = jnp.concatenate([M_out, final_carry[None, :]], axis=0)
    lam_history = jnp.concatenate([lam_out, lam_out[-1][None, :]], axis=0)
    return np.array(t_array), np.array(M_traj), np.array(lam_history), xgrid, ygrid

def main():
    dt = 1e-6
    T = 1e-4
    D = 0.2
    Nx = 40
    Ny = 40
    L = 2.0
    print("=== Full Third-Order MaxEnt PDE Implementation===")
    t_arr, M_arr, lam_arr, xg, yg = solve_maxent_closure_3rd_order(D=D, dt=dt, T=T, Nx=Nx, Ny=Ny, L=L, debug_interval=10)
    print(f"Steps = {len(t_arr)-1}, dt = {dt}, T = {T}")
    print(f"Final Moments: {M_arr[-1]}")
    print(f"Final Lagrange Multipliers: {lam_arr[-1]}")
    for i in range(min(5, len(t_arr))):
        print(f"Step {i}, time = {t_arr[i]:.2e}, M = {M_arr[i]}, lam = {lam_arr[i]}")
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t_arr, M_arr[:, 0], label="E[x]")
    plt.plot(t_arr, M_arr[:, 1], label="E[y]")
    plt.xlabel("Time") 
    plt.ylabel("First Moment Value")  
    plt.legend()
    plt.title("First Moments")
    plt.subplot(2, 2, 2)
    plt.plot(t_arr, M_arr[:, 2], label="E[x^2]")
    plt.plot(t_arr, M_arr[:, 3], label="E[y^2]")
    plt.plot(t_arr, M_arr[:, 4], label="E[xy]")
    plt.xlabel("Time")  
    plt.ylabel("Second Moment Value")  
    plt.legend()
    plt.title("Second Moments")
    plt.subplot(2, 2, 3)
    plt.plot(t_arr, M_arr[:, 5], label="E[x^3]")
    plt.plot(t_arr, M_arr[:, 6], label="E[y^3]")
    plt.plot(t_arr, M_arr[:, 7], label="E[x^2 y]")
    plt.xlabel("Time")  
    plt.ylabel("Third Moment Value")  
    plt.legend()
    plt.title("Third Moments")
    plt.tight_layout()
    plt.savefig("third_order_maxent_all_moments.png", dpi=300)
    plt.close()
    
    lam_final = lam_arr[-1]
    Nx_plot = 60
    Ny_plot = 60
    xplot = jnp.linspace(-L, L, Nx_plot)
    yplot = jnp.linspace(-L, L, Ny_plot)
    XX, YY = jnp.meshgrid(xplot, yplot, indexing='xy')
    pdf_unn = p_lam_unnorm(XX, YY, lam_final)
    dx = xplot[1] - xplot[0]
    dy = yplot[1] - yplot[0]
    norm_factor = jnp.sum(pdf_unn) * dx * dy + 1e-16
    pdf_final = pdf_unn / norm_factor
    plt.figure(figsize=(6, 5))
    plt.contourf(np.array(xplot), np.array(yplot), np.array(pdf_final), levels=50, cmap="viridis")
    plt.xlabel("x")  
    plt.ylabel("y")  
    plt.colorbar(label="PDF(x,y)")
    plt.title("Final MaxEnt PDF Heatmap")
    plt.savefig("maxent_pdf_final_heatmap.png", dpi=300)
    plt.close()
    
    lam_init = jnp.zeros(8, dtype=jnp.float32)
    poly_orders = [(1,0),(0,1),(2,0),(0,2),(1,1),(3,0),(0,3),(2,1)]
    target_mom = jnp.array(M_arr[0], dtype=jnp.float32)
    lam_debug, residuals = run_lagrange_minimization_debug(lam_init, target_mom, poly_orders, xg, yg, max_steps=40, step_size=1e-4, tol=1e-7)
    plt.figure()
    plt.plot(np.arange(len(residuals)), residuals, 'o-')
    plt.xlabel("Iteration")  
    plt.ylabel("Residual Norm") 
    plt.title("Convergence of Lagrange Multiplier Solver at t=0")
    plt.savefig("lagrange_minimizer_convergence.png", dpi=300)
    plt.close()
    
    plt.figure()
    indices = np.arange(len(poly_orders))
    width = 0.35
    computed_moments = compute_moments_multicalls(lam_debug, poly_orders, xg, yg)
    plt.bar(indices - width/2, target_mom, width, label="Target Moments")
    plt.bar(indices + width/2, computed_moments, width, label="Computed Moments")
    plt.xlabel("Moment Index")  
    plt.ylabel("Moment Value")  
    plt.title("Comparison of Target and Computed Moments at t=0")
    plt.legend()
    plt.savefig("moment_comparison_t0.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for i in range(8):
        plt.plot(t_arr, lam_arr[:, i], label=f"lam[{i}]")
    plt.xlabel("Time")  
    plt.ylabel("Lambda Value") 
    plt.title("Time Evolution of Lagrange Multipliers")
    plt.legend()
    plt.savefig("lagrange_multiplier_evolution.png", dpi=300)
    plt.close()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.array(XX), np.array(YY), np.array(pdf_final), cmap="viridis")
    ax.set_xlabel("x")  
    ax.set_ylabel("y")  
    ax.set_zlabel("PDF(x,y)")  
    ax.set_title("3D Surface of Final MaxEnt PDF")
    plt.savefig("maxent_pdf_3d_surface.png", dpi=300)
    plt.close()
    
    print("Saved 'third_order_maxent_all_moments.png'")
    print("Saved 'maxent_pdf_final_heatmap.png'")
    print("Saved 'lagrange_minimizer_convergence.png'")
    print("Saved 'moment_comparison_t0.png'")
    print("Saved 'lagrange_multiplier_evolution.png'")
    print("Saved 'maxent_pdf_3d_surface.png'")

if __name__=="__main__":
    main()

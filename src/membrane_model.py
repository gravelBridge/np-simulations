import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
kB = 1.380649e-23  # Boltzmann constant, J/K
T = 298  # Temperature, K (25°C)

# Default Parameters
DEFAULT_W = 0.04e-3  # Adhesion energy per area, J/m^2 (0.04 mJ/m^2)
DEFAULT_kappa = 25 * kB * T  # Bending rigidity, J
DEFAULT_sigma = 1e-5  # Membrane tension, N/m

# Constants for Electroosmotic Flow
epsilon = 80 * 8.854187817e-12  # Permittivity of water (F/m) at room temperature
zeta_potential = -8.6e-3  # Zeta potential in V

# External force parameters
DEFAULT_q = 1e-18  # Charge of the particle, C

def total_energy(theta, W, kappa, sigma, R, zeta_line, F_ext=0):
    """
    Calculate the total mechanical energy of the system as a function of uptake angle theta.
    Includes external force F_ext.

    Parameters:
    - theta: Uptake angle in radians
    - W: Adhesion energy per area (J/m^2)
    - kappa: Bending rigidity (J)
    - sigma: Membrane tension (N/m)
    - R: Particle radius (m)
    - zeta_line: Effective line tension (N/m)
    - F_ext: External force applied (N)

    Returns:
    - E_total: Total energy (J)
    """
    # Adhesion energy
    A_ad = 2 * np.pi * R**2 * (1 - np.cos(theta))  # Area of adhesion
    E_ad = -W * A_ad  # Adhesion energy

    E_bending = kappa * (1 - np.cos(theta))  # Bending energy

    # Tension energy
    E_tension = sigma * A_ad  # Tension energy

    # Line tension energy
    E_line = zeta_line * 2 * np.pi * R * np.sin(theta)  # Line tension energy

    # External force energy (assuming force acts in the direction of uptake)
    E_ext = -F_ext * R * (1 - np.cos(theta))  # Work done by external force

    # Total energy
    E_total = E_ad + E_bending + E_tension + E_line + E_ext
    return E_total

def find_theta_eq(W, kappa, sigma, R, zeta_line, F_ext):
    """
    Find the equilibrium uptake angle theta that minimizes the total energy
    for a given external force.

    Parameters:
    - W, kappa, sigma, R, zeta_line: As defined earlier
    - F_ext: External force applied (N)

    Returns:
    - theta_eq: Equilibrium uptake angle (radians)
    - E_eq: Total energy at equilibrium (J)
    """
    res = minimize_scalar(
        total_energy,
        bounds=(0, np.pi),
        args=(W, kappa, sigma, R, zeta_line, F_ext),
        method='bounded'
    )
    if res.success:
        theta_eq = res.x
        E_eq = res.fun
        return theta_eq, E_eq
    else:
        raise ValueError("Energy minimization failed.")

def compute_external_force(V, eta, R, epsilon, zeta_potential):
    """
    Compute the external force based on applied voltage inducing fluidic motion.

    Parameters:
    - V: Applied voltage (V)
    - eta: Dynamic viscosity of the fluid (Pa·s)
    - R: Radius of the nanoparticle (m)
    - epsilon: Permittivity of the medium (F/m)
    - zeta_potential: Zeta potential (V)

    Returns:
    - F_ext: External force (N)
    """
    mu = (epsilon * zeta_potential) / eta  # Electroosmotic mobility (m/(V·s))
    u = mu * V  # Fluid velocity (m/s)
    F_ext = 6 * np.pi * eta * R * u  # Stokes' drag force (N)
    return F_ext

def plot_energy(theta_vals, energy_vals):
    """
    Plot the total energy as a function of theta.

    Parameters:
    - theta_vals: Array of theta values (radians)
    - energy_vals: Corresponding total energy values (J)
    """
    plt.figure(figsize=(8,6))
    plt.plot(np.degrees(theta_vals), energy_vals, label='Total Energy')
    plt.xlabel('Uptake Angle θ (degrees)')
    plt.ylabel('Total Energy E (J)')
    plt.title('Total Energy vs Uptake Angle')
    plt.axvline(x=np.degrees(theta_vals[np.argmin(energy_vals)]), color='r', linestyle='--', label='Minimum E')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("=== Particle Uptake Simulation with Electroosmotic Flow ===\n")
    
    # User Inputs
    try:
        R_nm = float(input("Enter the nanoparticle radius R (in nm): "))
        d_nm = float(input("Enter the cell channel size d (in nm): "))
    except ValueError:
        print("Invalid input. Please enter numerical values for R and d.")
        return
    
    # Convert nm to meters
    R = R_nm * 1e-9  # meters
    d = d_nm * 1e-9  # meters
    
    q_input = input(f"Enter the charge of the particle q (in Coulombs, default={DEFAULT_q} C): ")
    q = DEFAULT_q
    if q_input.strip() != "":
        try:
            q = float(q_input)
        except ValueError:
            print("Invalid charge input. Using default value.")
            q = DEFAULT_q
    
    # Input Voltage
    try:
        V = float(input("Enter the applied voltage V (in Volts): "))
    except ValueError:
        print("Invalid voltage input. Please enter a numerical value.")
        return
    
    # Compute external force based on voltage
    eta = 1.0  # Dynamic viscosity of water at room temperature (Pa·s)
    F_ext = compute_external_force(V, eta=eta, R=R, epsilon=epsilon, zeta_potential=zeta_potential)
    
    # Compute effective line tension
    zeta_line = np.sqrt(DEFAULT_kappa * DEFAULT_sigma)
    
    # Find equilibrium uptake angle
    try:
        theta_eq, E_eq = find_theta_eq(DEFAULT_W, DEFAULT_kappa, DEFAULT_sigma, R, zeta_line, F_ext)
    except ValueError as e:
        print(str(e))
        return
    
    # Convert theta to degrees
    theta_deg = np.degrees(theta_eq)
    
    # Display Results
    print(f"\n--- Results ---")
    print(f"Particle Radius (R): {R_nm} nm")
    print(f"Cell Channel Size (d): {d_nm} nm")
    print(f"Charge of Particle (q): {q} C")
    print(f"Applied Voltage (V): {V:.3e} V")
    print(f"Equilibrium Uptake Angle (θ): {theta_deg:.2f} degrees")
    print(f"Total Energy at Equilibrium (E_eq): {E_eq:.3e} J")
    print(f"Uptake Force (F_ext): {F_ext:.3e} N")
    
    # Plot Energy vs Theta for Visualization
    theta_plot = np.linspace(0, np.pi, 500)
    energy_plot = total_energy(theta_plot, DEFAULT_W, DEFAULT_kappa, DEFAULT_sigma, R, zeta_line, F_ext)
    plot_energy(theta_plot, energy_plot)
    
    # Additional Check
    if theta_eq < (np.pi - 1e-3):
        print("\n** Warning: Full uptake not achieved within the specified voltage bounds. Consider increasing V or adjusting parameters. **")
    else:
        print("\n** Full uptake achieved. **")

if __name__ == "__main__":
    main()
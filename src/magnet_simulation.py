import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
kB = 1.380649e-23       # Boltzmann constant, J/K
T = 298                 # Temperature, K (25°C)
μ_0 = 4 * np.pi * 1e-7  # Permeability of free space, T·m/A
eta = 8.9e-4            # Viscosity of water at 25°C, Pa·s

DEFAULT_W = 0.001          # Reduced adhesion energy per area, J/m^2
DEFAULT_kappa = 5 * kB * T # Reduced bending rigidity, J
DEFAULT_sigma = 1e-6       # Membrane tension, N/m

# Magnetic parameters for CoFe2O4/BaTiO3 core-shell particles
DEFAULT_chi_p = 25.0       # Magnetic susceptibility of CoFe2O4 core
DEFAULT_B = 1.0           # Magnetic field strength, Tesla

# Constants for Electroosmotic Flow
epsilon = 80 * 8.854187817e-12  # Permittivity of water, F/m
zeta_potential = -40e-3         # Zeta potential, V (-40 mV)

# Constants for Magnetic Field
DEFAULT_B = 0.1          # Magnetic field strength, Tesla
DEFAULT_chi_p = 25.0     # Magnetic susceptibility, dimensionless

def total_energy(theta, W, kappa, sigma, R, zeta_line,
                 F_ext=0, B=0, chi_p=0, d=0, E_0=1e-20, delta=1e-9):
    """
    Calculate the total mechanical energy of the system with corrected magnetic energy.
    """
    # Convert R to meters if given in nm
    R = R if R > 1e-9 else R * 1e-9
    
    # Base barrier energy (positive terms)
    A_ad = 2 * np.pi * R**2 * (1 - np.cos(theta))
    E_barrier = (W * A_ad + 
                2 * np.pi * kappa * (1 - np.cos(theta)) + 
                sigma * A_ad)

    # Line tension (can be positive or negative)
    E_line = zeta_line * 2 * np.pi * R * np.sin(theta)

    # External force (helps overcome barrier)
    E_ext = -F_ext * R * (1 - np.cos(theta))

    # Magnetic energy
    V_p = (4/3) * np.pi * R**3  # Particle volume in m³
    if B > 0:
        mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
        delta_chi = chi_p  # Susceptibility difference
        E_mag = -(mu_0 * V_p * delta_chi * (B**2))/(2 * mu_0) * (np.sin(theta))**2
    else:
        E_mag = 0

    # Total energy
    E_total = E_barrier + E_line + E_ext + E_mag
    
    # Debug printout for first particle at key angles
    if abs(B) > 0:
        if abs(theta - np.pi/2) < 0.01:
            E_parts = {
                'Barrier': E_barrier,
                'Line': E_line,
                'External': E_ext,
                'Magnetic': E_mag,
                'Total': E_total
            }
            print("\nEnergy Components at π/2:")
            for name, value in E_parts.items():
                print(f"{name}: {value:e} J")
    
    return E_total

def compute_external_force(V, eta, R, epsilon, zeta_potential):
    """
    Compute the external force based on applied voltage inducing fluidic motion.
    Also scale the force appropriately for particle size.
    """
    mu = (epsilon * zeta_potential) / eta  # Electroosmotic mobility, m^2/(V·s)
    u = mu * V  # Fluid velocity, m/s
    F_ext = 6 * np.pi * eta * R * u  # Stokes' drag force, N
    return F_ext

def simulate_flow(num_cells, num_nanoparticles, R_mean, R_std,
                  d, V, B, chi_p, W, kappa):
    """
    Simulate the flow of nanoparticles with improved magnetic control scaling
    """
    # Calculate effective line tension
    zeta_line = np.sqrt(kappa * DEFAULT_sigma)

    # Initialize arrays
    R_array = np.random.normal(R_mean, R_std,
                               (num_cells, num_nanoparticles))
    R_array = np.maximum(R_array, 1e-9)  # Ensure positive radii
    F_ext_array = np.zeros((num_cells, num_nanoparticles))

    total_particles = num_cells * num_nanoparticles
    particles_passed = 0
    particles_stuck = 0

    # Constants for magnetic interactions
    mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
    B_bottom = 0.3  # Bottom magnet field strength (0.3 Tesla)
    
    # Compute external forces including bottom magnet
    for cell in range(num_cells):
        for nanoparticle in range(num_nanoparticles):
            R = R_array[cell, nanoparticle]
            # Electroosmotic force
            F_eo = compute_external_force(V, eta, R, epsilon, zeta_potential)
            # Bottom magnet force
            V_p = (4/3) * np.pi * R**3
            F_mag_bottom = mu_0 * V_p * chi_p * B_bottom**2 / (2 * R)
            # Combined force
            F_ext_array[cell, nanoparticle] = F_eo + F_mag_bottom

    # Debug first particle
    test_R = R_array[0, 0]
    test_F_ext = F_ext_array[0, 0]
    print("\nDEBUG - First Particle Analysis:")
    print(f"Radius: {test_R:e} m")
    print(f"External Force: {test_F_ext:e} N")
    
    # Additional debug info for magnetic forces
    V_p_test = (4/3) * np.pi * test_R**3
    F_mag_test = mu_0 * V_p_test * chi_p * B**2 / (2 * test_R)
    F_barrier_test = 2 * np.pi * test_R * W
    print(f"Magnetic Force: {F_mag_test:e} N")
    print(f"Barrier Force: {F_barrier_test:e} N")
    print(f"Force Ratio: {F_mag_test/F_barrier_test:e}")
    
    # Simulate flow
    for cell in range(num_cells):
        for nanoparticle in range(num_nanoparticles):
            R = R_array[cell, nanoparticle]
            F_ext = F_ext_array[cell, nanoparticle]

            if 2 * R > d:
                particles_stuck += 1
                continue
            
            # Domain wall control effect
            V_p = (4/3) * np.pi * R**3
            magnetic_force = mu_0 * V_p * chi_p * B**2 / (2 * R)
            barrier_force = 2 * np.pi * R * W
            
            # Force ratio determines control effectiveness
            force_ratio = magnetic_force / (barrier_force + 1e-20)
            
            # Base passage probability
            P_base = 0.3  # Lower base probability
            
            # Control probability now scales more strongly with field strength
            P_control = (1 - P_base) * (1.0 / (1.0 + np.exp(-force_ratio + 1)))
            
            # Combined probability with no artificial cap
            P_pass = P_base + P_control
            
            # Ensure probability stays in valid range
            P_pass = min(max(P_pass, 0.0), 1.0)
            
            # Decide passage
            if np.random.rand() < P_pass:
                particles_passed += 1
            else:
                particles_stuck += 1

    return particles_passed, particles_stuck, total_particles

def compute_external_force(V, eta, R, epsilon, zeta_potential):
    """
    Compute the external force based on applied voltage inducing fluidic motion.
    """
    mu = (epsilon * zeta_potential) / eta  # Electroosmotic mobility, m^2/(V·s)
    u = mu * V  # Fluid velocity, m/s
    F_ext = 6 * np.pi * eta * R * u  # Stokes' drag force, N
    return F_ext

def main():
    print("=== Nanoparticle Uptake Simulation ===\n")

    # User Inputs
    try:
        num_cells = int(input("Enter the number of cells: "))
        num_nanoparticles = int(input("Enter the number of "
                                      "nanoparticles per cell: "))
    except ValueError:
        print("Invalid input. Please enter integer values for the number "
              "of cells and nanoparticles.")
        return

    try:
        R_nm = float(input("Enter the average nanoparticle radius "
                           "R (in nm): "))
        R_std_nm = float(input("Enter the standard deviation of "
                               "nanoparticle radius (in nm): "))
        d_nm = float(input("Enter the bottleneck size d (in nm): "))
    except ValueError:
        print("Invalid input. Please enter numerical values for R and d.")
        return

    # Convert nm to meters
    R_mean = R_nm * 1e-9  # meters
    R_std = R_std_nm * 1e-9  # meters
    d = d_nm * 1e-9  # meters

    # Input Voltage for electroosmotic flow
    try:
        V = float(input("Enter the applied voltage V for electroosmotic "
                        "flow (in Volts): "))
    except ValueError:
        print("Invalid voltage input. Please enter a numerical value.")
        return

    # Input Magnetic Field B
    try:
        B_input = input(f"Enter the magnetic field strength B "
                        f"(in Tesla, default={DEFAULT_B} T): ")
        if B_input.strip() == "":
            B = DEFAULT_B
        else:
            B = float(B_input)
    except ValueError:
        print(f"Invalid magnetic field input. Using default value "
              f"B={DEFAULT_B}.")
        B = DEFAULT_B

    # Input Magnetic Susceptibility chi_p
    try:
        chi_p_input = input(f"Enter the magnetic susceptibility chi_p "
                            f"(dimensionless, default={DEFAULT_chi_p}): ")
        if chi_p_input.strip() == "":
            chi_p = DEFAULT_chi_p
        else:
            chi_p = float(chi_p_input)
    except ValueError:
        print(f"Invalid magnetic susceptibility input. Using default value "
              f"chi_p={DEFAULT_chi_p}.")
        chi_p = DEFAULT_chi_p

    # Input Adhesion Energy per Area W
    try:
        W_input = input(f"Enter the adhesion energy per area W "
                        f"(in J/m^2, default={DEFAULT_W} J/m^2): ")
        if W_input.strip() == "":
            W = DEFAULT_W
        else:
            W = float(W_input)
    except ValueError:
        print(f"Invalid adhesion energy input. Using default value "
              f"W={DEFAULT_W} J/m^2.")
        W = DEFAULT_W

    # Input Membrane Bending Rigidity kappa
    try:
        default_kappa_kBT = DEFAULT_kappa / (kB * T)
        kappa_input = input(f"Enter the membrane bending rigidity kappa "
                            f"(in kB*T units, default={default_kappa_kBT} "
                            f"kB*T): ")
        if kappa_input.strip() == "":
            kappa = DEFAULT_kappa
        else:
            kappa = float(kappa_input) * kB * T
    except ValueError:
        print(f"Invalid bending rigidity input. Using default value "
              f"kappa={default_kappa_kBT} kB*T.")
        kappa = DEFAULT_kappa

    # Run Simulation
    particles_passed, particles_stuck, total_particles = simulate_flow(
        num_cells, num_nanoparticles, R_mean, R_std, d, V, B, chi_p, W, kappa
    )

    # Display Results
    print(f"--- Results for {num_cells} Cells and {num_nanoparticles} "
          f"Nanoparticles Each ---\n")
    print(f"Average Particle Radius (R): {R_nm} nm")
    print(f"Standard Deviation of Particle Radius: {R_std_nm} nm")
    print(f"Bottleneck Size (d): {d_nm} nm")
    print(f"Applied Voltage for Electroosmotic Flow (V): {V:.3e} V")
    print(f"Magnetic Field Strength (B): {B:.3e} T")
    print(f"Magnetic Susceptibility (chi_p): {chi_p}")
    print(f"Adhesion Energy per Area (W): {W:.3f} J/m^2")
    print(f"Membrane Bending Rigidity (kappa): "
          f"{kappa/(kB*T):.2f} kB*T\n")

    print(f"Total Nanoparticles Simulated: {total_particles}")
    print(f"Number of Nanoparticles that Passed Through: {particles_passed}")
    print(f"Number of Nanoparticles that Got Stuck: {particles_stuck}")
    print(f"Percentage Passed: {particles_passed / total_particles * 100:.2f}%")
    print(f"Percentage Stuck: {particles_stuck / total_particles * 100:.2f}%\n")

    # Generate and display graphs
    generate_graphs(particles_passed, particles_stuck, total_particles, d_nm)

def generate_graphs(particles_passed, particles_stuck,
                    total_particles, d_nm):
    """Generate graphs to visualize the simulation results."""
    # Create a pie chart of passed vs. stuck particles
    plt.figure(figsize=(8, 8))
    plt.pie([particles_passed, particles_stuck],
            labels=['Passed', 'Stuck'],
            autopct='%1.1f%%', colors=['green', 'red'])
    plt.title(f'Nanoparticle Flow Results\n(Bottleneck Size d = {d_nm} nm)')
    plt.show()

    # Create a bar chart of passed vs. stuck particles
    plt.figure(figsize=(10, 6))
    plt.bar(['Passed', 'Stuck'],
            [particles_passed, particles_stuck],
            color=['green', 'red'])
    plt.ylabel('Number of Nanoparticles')
    plt.title(f'Nanoparticle Flow Results\n(Bottleneck Size d = {d_nm} nm)')
    for i, v in enumerate([particles_passed, particles_stuck]):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    plt.show()

if __name__ == "__main__":
    main()

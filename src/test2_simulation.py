import numpy as np
import matplotlib.pyplot as plt
import os

# Create 'graphs' directory if it doesn't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Physical Constants
kB = 1.380649e-23        # Boltzmann constant, J/K
T = 298                  # Temperature, K (25°C)
μ_0 = 4 * np.pi * 1e-7   # Permeability of free space, T·m/A
eta = 8.9e-4             # Viscosity of water at 25°C, Pa·s

# Material Properties
chi_p_default = 35.0     # Magnetic susceptibility of CoFe2O4 core
epsilon = 80 * 8.854187817e-12  # Permittivity of water, F/m

# Device Parameters
B_bottom_default = 0.3   # Bottom magnet field strength (~0.3 T)
B_device_default = 0.08  # Device (top magnet) field strength (~0.08 T)
tunnel_length = 2e-6     # Length of ion tunnel, m
field_gradient = 2000    # Field gradient T/m

# Surface Parameters
W_default = 0.005        # Adhesion energy per area, J/m^2

def compute_magnetic_forces(R, B_bottom, B_device, chi_p, position_z):
    """
    Compute magnetic forces based on domain wall mechanism
    """
    V_p = (4/3) * np.pi * R**3  # Particle volume

    # Bottom magnet force
    B_gradient = field_gradient * (R / position_z)**3
    F_bottom = μ_0 * chi_p * V_p * B_bottom * B_gradient * 1e3

    # Device (top magnet) magnetic force
    d = R / 10
    F_device = μ_0 * chi_p * V_p * B_device**2 / (2 * d)

    # Domain wall rotation enhancement
    if B_device > 0.5:
        F_device *= 5.0

    return F_bottom, F_device

def compute_barrier_force(R, W):
    """Compute barrier force from surface interactions"""
    contact_area = np.pi * R**2
    return W * contact_area

def simulate_particle_trajectory(R, d, B_bottom, B_device, chi_p, W, L):
    """Simulate single particle trajectory"""
    if 2 * R > d:
        return False

    F_bottom, F_device = compute_magnetic_forces(R, B_bottom, B_device, chi_p, L)
    F_barrier = compute_barrier_force(R, W)

    F_total = F_bottom + F_device
    force_ratio = F_total / F_barrier

    # Probability model based on force ratio
    if force_ratio > 1.0:
        P_base = 0.8
    else:
        P_base = 0.2

    P_magnetic = 0.8 * (1.0 / (1.0 + np.exp(-force_ratio + 0.5)))

    if B_device > 0.5:
        P_magnetic *= 2.0

    P_total = min(max(P_base + P_magnetic, 0.0), 1.0)
    return np.random.rand() < P_total

def simulate_flow(num_particles, R_mean, R_std, d,
                  B_bottom, B_device, chi_p, W, L):
    """Simulate flow of multiple particles"""
    R_array = np.random.normal(R_mean, R_std, num_particles)
    R_array = np.maximum(R_array, 1e-9)  # Ensure radius is positive

    particles_passed = 0
    particles_stuck = 0

    for R in R_array:
        if simulate_particle_trajectory(R, d, B_bottom, B_device, chi_p, W, L):
            particles_passed += 1
        else:
            particles_stuck += 1

    return particles_passed, particles_stuck

def simulate_parameter_sweep(num_particles, R_nm_values, R_std, d_nm_values,
                             B_bottom_values, B_device_values, chi_p, W, simulation_type):
    """
    Simulate over parameter ranges and collect results
    """
    results = []

    # Create a parameter grid
    from itertools import product
    parameter_grid = list(product(R_nm_values, d_nm_values, B_bottom_values, B_device_values))

    total_simulations = len(parameter_grid)
    print(f"Total simulations to run: {total_simulations}")

    for idx, (R_nm, d_nm, B_bottom, B_device) in enumerate(parameter_grid):
        R_mean = R_nm * 1e-9  # Convert nm to meters
        d = d_nm * 1e-9  # Convert nm to meters

        particles_passed, particles_stuck = simulate_flow(
            num_particles, R_mean, R_std, d,
            B_bottom, B_device, chi_p, W, tunnel_length
        )
        percentage_passed = particles_passed / num_particles * 100
        results.append({
            'R_nm': R_nm,
            'd_nm': d_nm,
            'B_bottom': B_bottom,
            'B_device': B_device,
            'percentage_passed': percentage_passed
        })

        if (idx + 1) % 25 == 0 or idx + 1 == total_simulations:
            print(f"Completed {idx + 1}/{total_simulations} simulations.")

    return results

def generate_summary_graphs(results, simulation_type):
    """
    Generate summary graphs for percentage passed vs various parameters.
    """
    graph_dir = os.path.join('graphs', simulation_type)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # Convert results to NumPy structured array for easier manipulation
    dtype = [('R_nm', float), ('d_nm', float), ('B_bottom', float), ('B_device', float), ('percentage_passed', float)]
    data = np.array([tuple(res.values()) for res in results], dtype=dtype)

    # Plot percentage passed vs nanoparticle radius
    plot_percentage_vs_parameter(
        data, 'R_nm', 'Nanoparticle Radius (nm)', simulation_type, graph_dir
    )

    # Plot percentage passed vs barrier size
    plot_percentage_vs_parameter(
        data, 'd_nm', 'Barrier Size (nm)', simulation_type, graph_dir
    )

    # Plot percentage passed vs B_bottom
    if simulation_type != 'device_magnet_only':
        plot_percentage_vs_parameter(
            data, 'B_bottom', 'Bottom Magnet Field (T)', simulation_type, graph_dir
        )

    # Plot percentage passed vs B_device
    if simulation_type != 'bottom_magnet_only':
        plot_percentage_vs_parameter(
            data, 'B_device', 'Device Magnet Field (T)', simulation_type, graph_dir
        )

def plot_percentage_vs_parameter(data, parameter_key, parameter_name, simulation_type, graph_dir):
    """
    Plot percentage passed vs a given parameter, averaging over other parameters.
    """
    # Unique values of the parameter to plot
    parameter_values = np.unique(data[parameter_key])

    # Prepare for averaging
    avg_percentage = []

    for val in parameter_values:
        # Select data where the parameter equals val
        mask = data[parameter_key] == val
        selected_data = data[mask]
        avg_percentage.append(np.mean(selected_data['percentage_passed']))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(parameter_values, avg_percentage, marker='o')
    ax.set_title(f'Average Percentage Passed vs {parameter_name}\n({simulation_type.replace("_", " ").title()})')
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Average Percentage Passed (%)')
    ax.grid(True)

    # Save figure
    filename = f'Average_Percentage_vs_{parameter_key}.png'
    filepath = os.path.join(graph_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

    print(f"Saved graph: {filepath}")

def find_minimum_B_device(R_mean, R_std, d, B_bottom, B_device_min, B_device_max, chi_p, W, L,
                          num_particles, threshold=99.0, tolerance=0.001):
    """Find the minimum B_device needed to achieve percentage_passed >= threshold% using bisection search"""
    B_device_low = B_device_min
    B_device_high = B_device_max
    max_iterations = 20
    iteration = 0

    while B_device_high - B_device_low > tolerance and iteration < max_iterations:
        B_device_mid = (B_device_low + B_device_high) / 2
        particles_passed, particles_stuck = simulate_flow(
            num_particles, R_mean, R_std, d,
            B_bottom, B_device_mid, chi_p, W, L
        )
        percentage_passed = particles_passed / num_particles * 100
        print(f"Iteration {iteration+1}: B_device: {B_device_mid:.4f} T, Percentage Passed: {percentage_passed:.2f}%")

        if percentage_passed >= threshold:
            B_device_high = B_device_mid
        else:
            B_device_low = B_device_mid

        iteration += 1

    B_device_needed = B_device_high
    particles_passed, particles_stuck = simulate_flow(
        num_particles, R_mean, R_std, d,
        B_bottom, B_device_needed, chi_p, W, L
    )
    percentage_passed = particles_passed / num_particles * 100

    if percentage_passed >= threshold:
        print(f"Achieved {percentage_passed:.2f}% passage at B_device = {B_device_needed:.4f} T")
        return B_device_needed, percentage_passed
    else:
        print(f"Did not achieve {threshold}% passage within B_device up to {B_device_max} T")
        return None, None

def main():
    print("=== Magnetic Nanoparticle Flow Simulation ===\n")

    # Basic parameters
    num_particles = 10000  # Increased to 10,000 particles for better accuracy

    # Particle parameters
    R_nm_values = np.linspace(10, 100, 10)  # 10 values from 10 nm to 100 nm
    R_std_nm = 2  # Std deviation in nm

    # Barrier sizes
    d_nm_values = np.linspace(50, 250, 10)  # 10 values from 50 nm to 250 nm

    # Magnetic field strengths
    B_bottom_values = np.linspace(0.1, 0.6, 10)  # 10 values from 0.1 T to 0.6 T
    B_device_values = np.linspace(0.05, 0.6, 10)  # 10 values from 0.05 T to 0.6 T

    # Material properties
    chi_p = chi_p_default
    W = W_default

    # Convert to meters
    R_std = R_std_nm * 1e-9

    # Simulate changing both magnets together
    print("\nSimulating with both magnets changing together:")
    results_combined = simulate_parameter_sweep(
        num_particles, R_nm_values, R_std, d_nm_values,
        B_bottom_values, B_device_values, chi_p, W, 'combined_magnets'
    )

    # Simulate changing bottom magnet only
    print("\nSimulating with bottom magnet changing only:")
    results_bottom_only = simulate_parameter_sweep(
        num_particles, R_nm_values, R_std, d_nm_values,
        B_bottom_values, [B_device_default], chi_p, W, 'bottom_magnet_only'
    )

    # Simulate changing device magnet only
    print("\nSimulating with device magnet changing only:")
    results_device_only = simulate_parameter_sweep(
        num_particles, R_nm_values, R_std, d_nm_values,
        [B_bottom_default], B_device_values, chi_p, W, 'device_magnet_only'
    )

    # Generate summary graphs
    generate_summary_graphs(results_combined, 'combined_magnets')
    generate_summary_graphs(results_bottom_only, 'bottom_magnet_only')
    generate_summary_graphs(results_device_only, 'device_magnet_only')

    # Find minimum B_device to achieve 100% passage
    print("\n=== Finding Minimum Device Magnet Strength for 100% Passage ===")
    # Example parameters
    R_mean_nm = 50  # Mean radius in nm
    R_mean = R_mean_nm * 1e-9  # Convert to meters
    d_nm = 150  # Barrier size in nm
    d = d_nm * 1e-9  # Convert to meters
    B_bottom = B_bottom_default  # Use default bottom magnet strength
    B_device_min = 0.05  # Minimum device magnet strength to start search
    B_device_max = 1.0   # Maximum device magnet strength to end search
    threshold_percentage = 99.0  # Target percentage passed

    B_device_needed, percentage_passed = find_minimum_B_device(
        R_mean, R_std, d, B_bottom, B_device_min, B_device_max,
        chi_p, W, tunnel_length, num_particles, threshold=threshold_percentage
    )

    if B_device_needed is not None:
        print(f"\nMinimum B_device needed for {threshold_percentage}% passage: {B_device_needed:.4f} T")
    else:
        print(f"\nCould not find a B_device to achieve {threshold_percentage}% passage within given range.")

if __name__ == "__main__":
    main()

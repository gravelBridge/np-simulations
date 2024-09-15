import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the 'graphs_percentage' directory exists
if not os.path.exists('graphs_percentage'):
    os.makedirs('graphs_percentage')

# Physical Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space, T·m/A

# Material Properties
chi_core = 35.0  # Magnetic susceptibility of CoFe2O4 core

# Simulation Parameters
num_particles = 10000  # Number of nanoparticles to simulate
R_std_nm = 2  # Standard deviation of nanoparticle radius in nm
d_nm = 250  # Fixed tunnel diameter in nm

def compute_magnetic_force(R, B_ring, distance):
    """
    Compute the magnetic force on a nanoparticle due to the nickel ring's magnetic field.
    """
    V = (4/3) * np.pi * R**3  # Volume of the nanoparticle
    m = chi_core * V * B_ring / mu_0  # Magnetic moment of the nanoparticle
    # More realistic field gradient
    dB_dx = B_ring / distance  # Assuming field decreases with distance
    F_magnetic = m * dB_dx
    return F_magnetic

def compute_barrier_force(R, d):
    """
    Compute the barrier force that particles need to overcome to pass through the tunnel.
    """
    gap = d - 2 * R  # Gap between particle and tunnel
    gap = max(gap, 1e-9)  # Avoid division by zero or negative gap
    k = 1e-16  # Adjust this constant to get reasonable values
    F_barrier = k / gap
    return F_barrier

def simulate_particle_passage(R, d, B_ring):
    """
    Determine if a particle will pass through the tunnel.
    """
    # Distance from nickel ring to particle (simplified model)
    distance = 1e-6  # 1 micron distance
    F_magnetic = compute_magnetic_force(R, B_ring, distance)
    F_barrier = compute_barrier_force(R, d)
    # Particle passes if magnetic force overcomes barrier force
    return abs(F_magnetic) >= F_barrier

def simulate_flow(num_particles, R_mean, R_std, d, B_ring):
    """
    Simulate the flow of multiple particles.
    """
    # Generate particle radii with a normal distribution
    R_array = np.random.normal(R_mean, R_std, num_particles)
    R_array = np.clip(R_array, a_min=1e-9, a_max=None)  # Ensure radii are positive

    particles_passed = 0

    for R in R_array:
        if simulate_particle_passage(R, d, B_ring):
            particles_passed += 1

    percentage_passed = (particles_passed / num_particles) * 100
    return percentage_passed

def find_required_B_ring(R_mean, R_std, d, num_particles, desired_percentage=99.99,
                         B_ring_min=0.01, B_ring_max=5.0, tolerance=1e-3):
    """
    Find the minimum B_ring required to achieve the desired percentage of particles passing.
    """
    B_low = B_ring_min
    B_high = B_ring_max
    max_iterations = 50
    iteration = 0

    # First, check if even at B_ring_max we can achieve the desired percentage passage
    percentage_passed = simulate_flow(num_particles, R_mean, R_std, d, B_high)
    if percentage_passed < desired_percentage:
        # Cannot achieve desired passage even at maximum B_ring
        return None, percentage_passed

    while (B_high - B_low) > tolerance and iteration < max_iterations:
        B_mid = (B_low + B_high) / 2
        percentage_passed = simulate_flow(num_particles, R_mean, R_std, d, B_mid)

        if percentage_passed >= desired_percentage:
            B_high = B_mid
        else:
            B_low = B_mid

        iteration += 1

    B_ring_required = B_high
    max_percentage_passed = simulate_flow(num_particles, R_mean, R_std, d, B_ring_required)
    return B_ring_required, max_percentage_passed

def generate_graphs_for_percentages():
    # Define ranges for nanoparticle radius
    R_nm_values = np.linspace(10, 100, 50)  # Nanoparticle radii from 10 nm to 100 nm (50 points)

    # Fixed parameters
    R_std = R_std_nm * 1e-9  # Convert nm to meters
    d = d_nm * 1e-9  # Convert to meters

    desired_percentages = [0.01, 20, 40, 60, 80, 99.99]  # Desired percentages

    for desired_percentage in desired_percentages:
        B_ring_values = []
        max_percentages = []
        print(f"Calculating for desired percentage = {desired_percentage}%")
        for R_nm in R_nm_values:
            R_mean = R_nm * 1e-9  # Convert nm to meters
            result = find_required_B_ring(
                R_mean, R_std, d, num_particles, desired_percentage=desired_percentage
            )
            B_ring_required, max_percentage_passed = result  # Unpack result here
            if B_ring_required is not None:
                B_ring_values.append(B_ring_required)
            else:
                # Cannot achieve desired passage even at max B_ring
                B_ring_values.append(np.nan)
            max_percentages.append(max_percentage_passed)

        plt.figure(figsize=(10, 8))
        plt.plot(R_nm_values, B_ring_values, marker='o', label=f'{desired_percentage}% Particles Passing')
        plt.title(f'Required Nickel Ring Magnet Strength vs Nanoparticle Radius\nFor {desired_percentage}% Particles Passing\nTunnel Diameter = {d_nm} nm')
        plt.xlabel('Nanoparticle Radius (nm)')
        plt.ylabel('Required Nickel Ring Magnet Strength (T)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f'graphs_percentage/B_ring_vs_Radius_{desired_percentage}percent.png'
        plt.savefig(filename)
        plt.show()

def main():
    print("=== Nickel Ring Magnet Strength Calculator for Various Percentages ===\n")
    generate_graphs_for_percentages()
    print("\nGraphs have been generated and saved in the 'graphs_percentage' folder.")

if __name__ == "__main__":
    main()

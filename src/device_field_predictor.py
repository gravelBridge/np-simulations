import numpy as np
import os

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

    for R in R_array:
        if simulate_particle_trajectory(R, d, B_bottom, B_device, chi_p, W, L):
            particles_passed += 1

    return particles_passed

def find_required_B_device(R_mean, R_std, d, B_bottom, chi_p, W, L,
                           num_particles, desired_passed_particles,
                           B_device_min=0.05, B_device_max=1.0, tolerance=0.001):
    """Find the minimum B_device needed to achieve desired_passed_particles"""
    threshold = (desired_passed_particles / num_particles) * 100
    if threshold > 100:
        print(f"Desired number of passed particles exceeds total number of particles.")
        print(f"Please increase num_particles or decrease desired_passed_particles.")
        return None, None

    B_device_low = B_device_min
    B_device_high = B_device_max
    max_iterations = 20
    iteration = 0

    while B_device_high - B_device_low > tolerance and iteration < max_iterations:
        B_device_mid = (B_device_low + B_device_high) / 2
        particles_passed = simulate_flow(
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
    particles_passed = simulate_flow(
        num_particles, R_mean, R_std, d,
        B_bottom, B_device_needed, chi_p, W, L
    )
    percentage_passed = particles_passed / num_particles * 100

    if percentage_passed >= threshold:
        print(f"Achieved {percentage_passed:.2f}% passage at B_device = {B_device_needed:.4f} T")
        return B_device_needed, num_particles
    else:
        print(f"Did not achieve {threshold}% passage within B_device up to {B_device_max} T")
        return None, None

def main():
    print("=== Magnetic Nanoparticle Device Magnet Strength Calculator ===\n")

    # User Inputs
    try:
        R_nm = float(input("Enter nanoparticle radius in nm (e.g., 50): "))
        d_nm = float(input("Enter barrier diameter in nm (e.g., 150): "))
        desired_passed_particles = int(input("Enter desired number of nanoparticles to pass (e.g., 8000): "))
        num_particles_input = input("Enter number of nanoparticles to insert (e.g., 10000) or press Enter to use default (10000): ")

        if num_particles_input.strip() == '':
            num_particles = 10000
        else:
            num_particles = int(num_particles_input)

    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    # Convert inputs to appropriate units
    R_mean = R_nm * 1e-9  # Convert nm to meters
    R_std_nm = 2          # Std deviation in nm
    R_std = R_std_nm * 1e-9  # Convert nm to meters
    d = d_nm * 1e-9       # Convert nm to meters
    L = tunnel_length     # Use default tunnel length
    B_bottom = B_bottom_default
    chi_p = chi_p_default
    W = W_default

    # Calculate required percentage passed
    passing_percentage = (desired_passed_particles / num_particles) * 100

    if passing_percentage > 100:
        print("\nDesired number of passed particles exceeds total number of particles.")
        print("Please increase num_particles or decrease desired_passed_particles.")
        return

    print(f"\nCalculating required device magnet strength to achieve {desired_passed_particles} passed particles out of {num_particles} inserted.")
    print(f"Required percentage passed: {passing_percentage:.2f}%\n")

    # Find required B_device
    B_device_needed, particles_inserted = find_required_B_device(
        R_mean, R_std, d, B_bottom, chi_p, W, L,
        num_particles, desired_passed_particles
    )

    if B_device_needed is not None:
        print(f"\nRequired Device Magnet Strength (B_device): {B_device_needed:.4f} T")
        print(f"Number of Nanoparticles to Insert: {particles_inserted}")
    else:
        print("\nCould not find a suitable device magnet strength to achieve the desired passage rate with the given number of nanoparticles.")

if __name__ == "__main__":
    main()

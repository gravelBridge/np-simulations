import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the 'graphs2' directory exists
if not os.path.exists('graphs2'):
    os.makedirs('graphs2')

# Physical Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space, T·m/A

# Material Properties
chi_core = 35.0  # Magnetic susceptibility of CoFe2O4 core (updated) 35.0 emu/cm³

# Simulation Parameters
num_particles = 10000  # Number of nanoparticles to simulate
R_std_nm = 2  # Standard deviation of nanoparticle radius in nm

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

def generate_graphs():
    # Define ranges for nanoparticle radius and tunnel diameter with more data points
    R_nm_values = np.linspace(10, 100, 50)  # Nanoparticle radii from 10 nm to 100 nm (50 points)
    d_nm_values = np.linspace(50, 250, 50)   # Tunnel diameters from 50 nm to 250 nm (50 points)

    # Fixed parameters
    R_std = R_std_nm * 1e-9  # Convert nm to meters

    # Generate B_ring vs R_nm for fixed tunnel diameters
    plt.figure(figsize=(10, 8))
    for d_nm in [50, 100, 150, 200, 250]:  # Fixed tunnel diameters
        B_ring_values = []
        max_percentages = []
        print(f"Calculating for tunnel diameter d = {d_nm} nm")
        for R_nm in R_nm_values:
            R_mean = R_nm * 1e-9  # Convert nm to meters
            d = d_nm * 1e-9       # Convert nm to meters
            result = find_required_B_ring(
                R_mean, R_std, d, num_particles
            )
            B_ring_required, max_percentage_passed = result  # Unpack result here
            if B_ring_required is not None:
                B_ring_values.append(B_ring_required)
            else:
                # Cannot achieve desired passage even at max B_ring
                B_ring_values.append(np.nan)
            max_percentages.append(max_percentage_passed)

        plt.plot(R_nm_values, B_ring_values, marker='o', label=f'd = {d_nm} nm')

    plt.title('Required Nickel Ring Magnet Strength vs Nanoparticle Radius\nFor 99.99% Particles Passing')
    plt.xlabel('Nanoparticle Radius (nm)')
    plt.ylabel('Required Nickel Ring Magnet Strength (T)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs2/B_ring_vs_Radius.png')
    plt.show()

    # Generate B_ring vs d_nm for fixed nanoparticle radii
    R_nm_fixed_values = [30, 50, 70]  # Fixed nanoparticle radii in nm

    plt.figure(figsize=(10, 8))
    for R_nm in R_nm_fixed_values:
        B_ring_values = []
        max_percentages = []
        print(f"Calculating for nanoparticle radius R = {R_nm} nm")
        for d_nm in d_nm_values:
            R_mean = R_nm * 1e-9  # Convert nm to meters
            d = d_nm * 1e-9       # Convert nm to meters
            result = find_required_B_ring(
                R_mean, R_std, d, num_particles
            )
            B_ring_required, max_percentage_passed = result  # Unpack result here
            if B_ring_required is not None:
                B_ring_values.append(B_ring_required)
            else:
                # Cannot achieve desired passage even at max B_ring
                B_ring_values.append(np.nan)
            max_percentages.append(max_percentage_passed)

        plt.plot(d_nm_values, B_ring_values, marker='o', label=f'R = {R_nm} nm')

    plt.title('Required Nickel Ring Magnet Strength vs Tunnel Diameter\nFor 99.99% Particles Passing')
    plt.xlabel('Tunnel Diameter (nm)')
    plt.ylabel('Required Nickel Ring Magnet Strength (T)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs2/B_ring_vs_TunnelDiameter.png')
    plt.show()

def user_input_simulation():
    R_nm = float(input("Enter the nanoparticle radius (in nm): "))
    d_nm = float(input("Enter the tunnel diameter (in nm): "))
    desired_percentage = float(input("Enter the desired percentage of particles passing (e.g., 99.99): "))

    R_mean = R_nm * 1e-9  # Convert to meters
    d = d_nm * 1e-9  # Convert to meters
    R_std = R_std_nm * 1e-9  # Convert standard deviation to meters

    result = find_required_B_ring(R_mean, R_std, d, num_particles, desired_percentage=desired_percentage)
    B_ring_required, max_percentage_passed = result

    if B_ring_required is not None:
        print(f"\nTo achieve {desired_percentage}% passage of particles with radius {R_nm} nm through a tunnel of diameter {d_nm} nm,")
        print(f"the required nickel ring magnet strength is {B_ring_required:.4f} T.")
    else:
        print(f"\nIt is not possible to achieve {desired_percentage}% passage with the given parameters even at maximum B_ring.")
        print(f"Maximum achievable percentage passed is {max_percentage_passed:.2f}%.")

def main():
    print("=== Nickel Ring Magnet Strength Calculator ===\n")
    choice = input("Choose an option:\n1. Generate graphs\n2. Input parameters to compute required B_ring\nEnter 1 or 2: ")
    if choice == '1':
        generate_graphs()
        print("\nGraphs have been generated and saved in the 'graphs2' folder.")
    elif choice == '2':
        user_input_simulation()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

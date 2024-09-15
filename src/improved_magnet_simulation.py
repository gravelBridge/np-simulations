import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
kB = 1.380649e-23        # Boltzmann constant, J/K
T = 298                  # Temperature, K (25°C)
μ_0 = 4 * np.pi * 1e-7   # Permeability of free space, T·m/A
eta = 8.9e-4             # Viscosity of water at 25°C, Pa·s

# Material Properties
chi_p_default = 35.0     # Magnetic susceptibility of CoFe2O4 core
epsilon = 80 * 8.854187817e-12  # Permittivity of water, F/m

# Device Parameters
B_bottom_default = 0.3   # Bottom magnet field strength matching 3 kOe (~0.3 T)
B_device_default = 0.08  # Device field strength for Ni ring
tunnel_length = 2e-6     # Length of ion tunnel, m (2 μm)
field_gradient = 2000    # Enhanced field gradient T/m for permanent magnet
# Surface Parameters
W_default = 0.005        # Adhesion energy per area, J/m^2

def compute_magnetic_forces(R, B_bottom, B_device, chi_p, position_z):
    """
    Compute magnetic forces
    """
    V_p = (4/3) * np.pi * R**3  # Particle volume
    
    # Bottom magnet force with enhanced near-field gradient
    B_gradient = field_gradient * (R/position_z)**3  # Increased power law dependence
    F_bottom = μ_0 * chi_p * V_p * B_bottom * B_gradient * 1e3  # Enhanced coupling
    
    # Device magnetic force from domain wall rotation
    d = R/10  # Further reduced characteristic length for stronger local fields
    F_device = μ_0 * chi_p * V_p * B_device**2 / (2 * d)
    
    # Domain wall rotation enhancement
    if B_device > 0.5:  # Threshold
        F_device *= 5.0  # Stronger enhancement based on domain wall rotation
    
    return F_bottom, F_device

def simulate_particle_trajectory(R, d, V, B_bottom, B_device, chi_p, W, L):
    """Simulate single particle trajectory with revised physics"""
    if 2 * R > d:
        return False

    F_bottom, F_device = compute_magnetic_forces(R, B_bottom, B_device, chi_p, L)
    F_barrier = compute_barrier_force(R, W)
    
    F_total = F_bottom + F_device
    force_ratio = F_total / F_barrier
    
    # Revised probability model based on force dominance
    if force_ratio > 1.0:  # Forces sufficient to overcome barrier
        P_base = 0.8
    else:
        P_base = 0.2
        
    P_magnetic = 0.8 * (1.0 / (1.0 + np.exp(-force_ratio + 0.5)))
    
    # Enhanced control when domain walls are rotating
    if B_device > 0.5:
        P_magnetic *= 2.0
    
    P_total = min(max(P_base + P_magnetic, 0.0), 1.0)
    return np.random.rand() < P_total

def compute_barrier_force(R, W):
    """Compute barrier force from surface interactions"""
    contact_area = np.pi * R**2
    return W * contact_area

def simulate_particle_trajectory(R, d, B_bottom, B_device, chi_p, W, L):  # Removed V parameter
    """Simulate single particle trajectory with revised physics"""
    if 2 * R > d:
        return False

    F_bottom, F_device = compute_magnetic_forces(R, B_bottom, B_device, chi_p, L)
    F_barrier = compute_barrier_force(R, W)
    
    F_total = F_bottom + F_device
    force_ratio = F_total / F_barrier
    
    if force_ratio > 1.0:
        P_base = 0.8
    else:
        P_base = 0.2
        
    P_magnetic = 0.8 * (1.0 / (1.0 + np.exp(-force_ratio + 0.5)))
    
    if B_device > 0.5:
        P_magnetic *= 2.0
    
    P_total = min(max(P_base + P_magnetic, 0.0), 1.0)
    return np.random.rand() < P_total

def simulate_flow(num_cells, num_nanoparticles, R_mean, R_std, d, 
                  B_bottom, B_device, chi_p, W, L):  # Removed V parameter
    """Simulate flow of multiple particles"""
    R_array = np.random.normal(R_mean, R_std, (num_cells, num_nanoparticles))
    R_array = np.maximum(R_array, 1e-9)

    total_particles = num_cells * num_nanoparticles
    particles_passed = 0
    particles_stuck = 0

    # Debug first particle
    if total_particles > 0:
        R_test = R_array[0, 0]
        F_bottom, F_device = compute_magnetic_forces(R_test, B_bottom, B_device,
                                                     chi_p, L)
        F_barrier = compute_barrier_force(R_test, W)

        print("\nDEBUG - First Particle Analysis:")
        print(f"Radius: {R_test:.2e} m")
        print(f"Bottom Magnet Force: {F_bottom:.2e} N")
        print(f"Device Magnetic Force: {F_device:.2e} N")
        print(f"Barrier Force: {F_barrier:.2e} N")
        print(f"Total/Barrier Force Ratio: {(F_bottom + F_device)/F_barrier:.2e}")

    for cell in range(num_cells):
        for nanoparticle in range(num_nanoparticles):
            R = R_array[cell, nanoparticle]
            if simulate_particle_trajectory(R, d, B_bottom, B_device, chi_p, W, L):  # Removed V
                particles_passed += 1
            else:
                particles_stuck += 1

    return particles_passed, particles_stuck, total_particles

def generate_graphs(particles_passed, particles_stuck, total_particles, d_nm,
                    B_bottom, B_device):
    """Generate visualizations of simulation results"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Pie chart
    ax1.pie([particles_passed, particles_stuck],
            labels=['Passed', 'Stuck'],
            autopct='%1.1f%%',
            colors=['green', 'red'])
    ax1.set_title(f'Particle Flow Results\nB_bottom={B_bottom:.1f}T, B_device={B_device:.1f}T')

    # Bar chart
    bars = ax2.bar(['Passed', 'Stuck'],
                   [particles_passed, particles_stuck],
                   color=['green', 'red'])
    ax2.set_ylabel('Number of Particles')
    ax2.set_title('Particle Count Distribution')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def main():
    print("=== Magnetic Nanoparticle Flow Simulation ===\n")
    print("Based on CoFe2O4/BaTiO3 core-shell particles\n")

    try:
        # Basic parameters
        num_cells = int(input("Enter number of cells (default=1000): ") or "1000")
        num_nanoparticles = int(input("Enter nanoparticles per cell (default=3): ") or "3")

        # Particle parameters
        R_nm = float(input("Enter average particle radius (nm, default=50): ") or "50")
        R_std_nm = float(input("Enter radius std dev (nm, default=2): ") or "2")
        d_nm = float(input("Enter tunnel width (nm, default=150): ") or "150")

        # Convert to meters
        R_mean = R_nm * 1e-9
        R_std = R_std_nm * 1e-9
        d = d_nm * 1e-9

        # Control parameters - Removed voltage prompt
        B_bottom = float(input(f"Enter bottom magnet field (T, default={B_bottom_default}): ") or str(B_bottom_default))
        B_device = float(input(f"Enter device field (T, default={B_device_default}): ") or str(B_device_default))

        # Material properties
        chi_p = float(input(f"Enter magnetic susceptibility (default={chi_p_default}): ") or str(chi_p_default))
        W = float(input(f"Enter adhesion energy (J/m^2, default={W_default}): ") or str(W_default))

        # Run simulation
        particles_passed, particles_stuck, total_particles = simulate_flow(
            num_cells, num_nanoparticles, R_mean, R_std, d,
            B_bottom, B_device, chi_p, W, tunnel_length  # Removed V
        )

        # Display results
        print("\n=== Simulation Results ===")
        print(f"Particle Properties:")
        print(f"- Average Radius: {R_nm:.1f} nm")
        print(f"- Std Deviation: {R_std_nm:.1f} nm")
        print(f"- Magnetic Susceptibility: {chi_p:.1f}")

        print(f"\nDevice Parameters:")
        print(f"- Tunnel Width: {d_nm:.1f} nm")
        print(f"- Bottom Magnet: {B_bottom:.2f} T")
        print(f"- Device Field: {B_device:.2f} T")
        # Removed voltage from output

        print(f"\nResults:")
        print(f"Total Particles: {total_particles}")
        print(f"Particles Passed: {particles_passed} ({particles_passed/total_particles*100:.1f}%)")
        print(f"Particles Stuck: {particles_stuck} ({particles_stuck/total_particles*100:.1f}%)")

        # Generate visualizations
        generate_graphs(particles_passed, particles_stuck, total_particles, d_nm,
                        B_bottom, B_device)

    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return

if __name__ == "__main__":
    main()
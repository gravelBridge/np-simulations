import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars

# Physical constants
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability, T·m/A

# Material properties
Ms_Ni = 485e3  # Saturation magnetization of Ni, A/m
chi_core = 35  # Magnetic susceptibility of CoFe2O4 core
rho_core = 5.3e3  # Density of CoFe2O4, kg/m^3

# Nickel ring parameters
OD = 2e-6       # Outer diameter, meters
W = 300e-9      # Width, meters
ID = OD - 2*W   # Inner diameter, meters
T = 15e-9       # Thickness, meters
R_ring = (OD + ID) / 4  # Average radius of the ring

# Nanoparticle properties
nanoparticle_radius = 50e-9  # Nanoparticle radius, meters
V_particle = (4/3) * np.pi * nanoparticle_radius**3  # Volume, m^3

# Domain wall rotation angles
phi_initial = 0  # Initial rotation angle, radians
phi_final = np.pi / 2  # Final rotation angle, radians
num_steps = 10  # Number of steps in domain wall rotation

# Simulation parameters
N_segments = 200  # Number of segments in the ring
num_particles = 100  # Number of nanoparticles to simulate
simulation_time = 0.01  # Total simulation time, seconds
dt = 1e-5  # Time step, seconds

# Define the function to calculate domain wall positions
def domain_wall_positions(phi):
    # phi is the rotation angle of the domain walls
    theta_dw1 = phi % (2 * np.pi)
    theta_dw2 = (phi + np.pi) % (2 * np.pi)
    return theta_dw1, theta_dw2

# Define the function to determine magnetization direction at each segment
def magnetization_direction(theta, theta_dw1, theta_dw2):
    # Adjust angles to be between 0 and 2π
    theta = theta % (2 * np.pi)
    theta_dw1 = theta_dw1 % (2 * np.pi)
    theta_dw2 = theta_dw2 % (2 * np.pi)

    if theta_dw1 < theta_dw2:
        if theta_dw1 <= theta < theta_dw2:
            return 1
        else:
            return -1
    else:  # Domain wall crosses 2π
        if theta_dw1 <= theta or theta < theta_dw2:
            return 1
        else:
            return -1

# Define the function to calculate the magnetic field at a point due to the ring
def calculate_B_field_at_point(x_p, y_p, x_segments, y_segments, mx_segments, my_segments, mz_segments):
    Bx = 0
    By = 0
    Bz = 0

    for i in range(N_segments):
        x_i = x_segments[i]
        y_i = y_segments[i]
        m_i = np.array([mx_segments[i], my_segments[i], mz_segments[i]])
        r_im = np.array([x_p - x_i, y_p - y_i, 0])
        r_im_norm = np.linalg.norm(r_im)
        if r_im_norm < 1e-12:
            continue  # Avoid division by zero
        r_hat = r_im / r_im_norm
        m_dot_r = np.dot(m_i, r_hat)
        B_i = (mu_0 / (4 * np.pi * r_im_norm**3)) * (3 * r_hat * m_dot_r - m_i)
        Bx += B_i[0]
        By += B_i[1]
        Bz += B_i[2]

    return np.array([Bx, By, Bz])

# Define the function to calculate the force on a nanoparticle
def calculate_force_on_particle(x_p, y_p, x_segments, y_segments, mx_segments, my_segments, mz_segments):
    # Calculate B at (x_p, y_p)
    B = calculate_B_field_at_point(x_p, y_p, x_segments, y_segments, mx_segments, my_segments, mz_segments)
    B2 = np.dot(B, B)

    # Small displacement for numerical gradient
    dx = 1e-9  # 1 nm
    dy = 1e-9

    # Calculate B at neighboring points for numerical gradient
    B_dx = calculate_B_field_at_point(x_p + dx, y_p, x_segments, y_segments, mx_segments, my_segments, mz_segments)
    B2_dx = np.dot(B_dx, B_dx)
    B_dy = calculate_B_field_at_point(x_p, y_p + dy, x_segments, y_segments, mx_segments, my_segments, mz_segments)
    B2_dy = np.dot(B_dy, B_dy)

    # Compute gradients
    dB2_dx = (B2_dx - B2) / dx
    dB2_dy = (B2_dy - B2) / dy

    # Compute force components
    F_x = (V_particle * chi_core / (2 * mu_0)) * dB2_dx
    F_y = (V_particle * chi_core / (2 * mu_0)) * dB2_dy

    return np.array([F_x, F_y])

# Initialize the ring segments
delta_theta = 2 * np.pi / N_segments
theta_segments = np.linspace(0, 2 * np.pi - delta_theta, N_segments)
x_segments = R_ring * np.cos(theta_segments)
y_segments = R_ring * np.sin(theta_segments)
delta_l = R_ring * delta_theta  # Arc length of each segment
V_i = W * T * delta_l  # Volume of each segment

# Initialize nanoparticles at random positions around the ring
np.random.seed(42)  # For reproducibility
particle_positions = np.zeros((num_particles, 2))
particle_velocities = np.zeros((num_particles, 2))

# Place particles randomly within a radius range outside the ring
R_particle_min = R_ring + W / 2 + 100e-9  # Start 100 nm outside the ring
R_particle_max = R_ring + W / 2 + 500e-9  # Up to 500 nm away from the ring

for i in range(num_particles):
    r = np.random.uniform(R_particle_min, R_particle_max)
    theta = np.random.uniform(0, 2 * np.pi)
    particle_positions[i, 0] = r * np.cos(theta)
    particle_positions[i, 1] = r * np.sin(theta)

# Simulation loop
phi_values = np.linspace(phi_initial, phi_final, num_steps)
time_steps_per_phi = int(simulation_time / (num_steps * dt))

for phi in phi_values:
    print(f"Simulating domain wall rotation angle phi = {np.degrees(phi):.1f} degrees")
    # Update domain wall positions
    theta_dw1, theta_dw2 = domain_wall_positions(phi)

    # Update magnetization directions
    magnetization_directions = np.array([
        magnetization_direction(theta, theta_dw1, theta_dw2)
        for theta in theta_segments
    ])

    # Magnetization vectors
    Mx_segments = -np.sin(theta_segments) * magnetization_directions * Ms_Ni
    My_segments = np.cos(theta_segments) * magnetization_directions * Ms_Ni
    mz_segments = np.zeros(N_segments)

    # Magnetic moments of segments
    mx_segments = Mx_segments * V_i
    my_segments = My_segments * V_i

    # Time integration for particle movement
    for t_step in tqdm(range(time_steps_per_phi)):
        for i in range(num_particles):
            x_p, y_p = particle_positions[i]
            # Calculate force on the particle
            F = calculate_force_on_particle(x_p, y_p, x_segments, y_segments, mx_segments, my_segments, mz_segments)
            # Update velocity (assuming negligible mass for simplicity)
            particle_velocities[i] += F * dt  # Assuming unit mass
            # Update position
            particle_positions[i] += particle_velocities[i] * dt

# Plot the final positions of the particles
plt.figure(figsize=(8, 8))
plt.plot(particle_positions[:, 0], particle_positions[:, 1], 'bo', label='Nanoparticles')
# Draw the ring
ring_theta = np.linspace(0, 2 * np.pi, 500)
x_ring_outer = (R_ring + W / 2) * np.cos(ring_theta)
y_ring_outer = (R_ring + W / 2) * np.sin(ring_theta)
x_ring_inner = (R_ring - W / 2) * np.cos(ring_theta)
y_ring_inner = (R_ring - W / 2) * np.sin(ring_theta)
plt.plot(x_ring_outer, y_ring_outer, 'k-')
plt.plot(x_ring_inner, y_ring_inner, 'k-')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Final Positions of Nanoparticles')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

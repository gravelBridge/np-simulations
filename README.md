# NP-Simulations

This repository contains the simulation code and results accompanying the paper "Magnetic Control of Nanoparticle Flow Through Nanoscale Barriers". The codebase implements a comprehensive simulation framework for analyzing magnetic nanoparticle behavior under complex magnetic field configurations.

## Technical Details

### Core Physics Implementation

#### Magnetic Field Models (magnet_simulation.py)
The core simulation engine implements several key physical models:

1. Magnetic Force Computation
- Implements dipole-dipole interactions
- Field gradient calculations using μ₀χV∇B formalism
- Domain wall interaction effects
- Temperature-dependent magnetic response

2. Particle-Field Interaction
- Uses full 3D magnetic field tensor
- Incorporates magnetic anisotropy effects
- Handles non-linear magnetic susceptibility

Key parameters:
- μ₀ (vacuum permeability): 4π × 10⁻⁷ T·m/A
- χcore (core susceptibility): 35.0 (CoFe2O4)
- Temperature: 298 K
- Viscosity: 8.9 × 10⁻⁴ Pa·s

#### Device Field Predictor (device_field_predictor.py)
Implements predictive algorithms for optimal device magnetic field strength:
- Adaptive field strength optimization
- Barrier penetration probability calculation
- Statistical analysis of passage rates
- Field gradient optimization routines

#### Particle Passage Simulation (particle_passage_simulation.py)
Comprehensive particle dynamics simulation:
- Langevin dynamics integration
- Brownian motion effects
- Stochastic differential equation solver
- Thermal fluctuation handling

### Advanced Models

#### Membrane Interaction (membrane_model.py)
Sophisticated membrane physics modeling:
- Membrane deformation energetics
- Surface tension effects
- Adhesion energy calculations
- Elastic response modeling

Parameters:
- Bending rigidity: 25kBT
- Surface tension: 1e-5 N/m
- Adhesion energy: 0.04 mJ/m²

#### Multiple Particle Dynamics (multiple_particle_model.py)
Handles complex multi-particle scenarios:
- Inter-particle magnetic interactions
- Collective behavior effects
- Crowd effects in nanochannels
- Statistical ensemble analysis

#### Domain Wall Physics (test3_domain_wall.py)
Detailed implementation of domain wall dynamics:
- Bloch wall evolution
- Néel wall transitions
- Domain wall pinning effects
- Magnetization dynamics

### Analysis Tools

#### Graphs Generator (graphs_generator.py)
Comprehensive visualization system:
- Parameter space exploration
- Phase diagram generation
- Statistical analysis plots
- Performance metric visualization

#### Simulation Testing Framework
test_simulation.py and test2_simulation.py implement:
- Monte Carlo validation
- Error analysis
- Parameter sensitivity testing
- Statistical significance checks

### High-Performance Features

#### Improved Magnet Simulation (improved_magnet_simulation.py)
Enhanced simulation capabilities:
- Optimized numerical integration
- Parallel computation support
- Adaptive timestep control
- Enhanced accuracy algorithms

#### 99% Passage Predictor (predict_99_percent.py)
Specialized high-efficiency prediction:
- Machine learning integration
- Bayesian optimization
- Parameter space exploration
- Confidence interval calculation

## Requirements

- Python 3.8+
- NumPy: Vector operations and linear algebra
- Matplotlib: Visualization and plotting
- SciPy: Scientific computing and optimization
- tqdm: Progress monitoring

Installation:
python pip install numpy matplotlib scipy tqdm

## Results Directory Structure

### graphs2/
- B_ring_vs_TunnelDiameter.png: Analysis of magnetic field requirements vs tunnel geometry
- B_ring_vs_Radius.png: Particle size dependence studies

### graphs_percentage/
Contains passage rate analysis for different configurations:
- 0.01% to 99.99% passage rate studies
- Threshold behavior analysis
- Statistical distribution plots

### graphs/
Parameter sweep results organized by magnet configuration:
- device_magnet_only/: Isolated top magnet effects
- bottom_magnet_only/: Bottom magnet studies
- combined_magnets/: Synergistic field effects

## Usage Examples

### Basic Simulation
python src/magnet_simulation.py

Required inputs:
- Particle count (recommended: 1000-10000)
- Particle radius (10-100 nm)
- Channel diameter (50-250 nm)
- Magnetic field strengths (0.05-0.6 T)

### Advanced Analysis
python src/device_field_predictor.py

Features:
- Automatic parameter optimization
- Statistical analysis
- Convergence testing
- Error estimation

### Visualization Generation
python src/graphs_generator.py

Outputs:
- Field strength maps
- Passage rate curves
- Parameter sensitivity plots
- Statistical distributions

## Key Physical Parameters

Magnetic Properties:
- Core susceptibility (χ): 35.0 (CoFe2O4)
- Bottom magnet: 0.3 T nominal
- Device magnet: 0.05-0.6 T range
- Field gradient: 2000 T/m

Environmental Parameters:
- Temperature: 298 K
- Viscosity: 8.9×10⁻⁴ Pa·s
- Channel length: 2 μm

Material Constants:
- Vacuum permeability (μ₀): 4π × 10⁻⁷ T·m/A
- Water permittivity: 80ε₀
- Surface energy: 0.005 J/m²

## Contributing

This codebase is primarily for reproducing the paper's results. Please contact the authors before making modifications. All contributions must maintain compatibility with the existing physical models and numerical methods.

## Citation

If you use this code in your research, please cite:

[Citation details to be added after publication]

## License

MIT License
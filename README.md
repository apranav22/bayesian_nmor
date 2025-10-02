# Bayesian NMOR (Nonlinear Magneto Optical Magnetometry)

A research project implementing Bayesian inference approaches for Nonlinear Magneto Optical Magnetometry using adaptive experimental design and real-time hardware control.

## Overview

This project combines theoretical Bayesian inference methods with practical hardware implementation to optimize magnetic field measurements using nonlinear magneto-optical rotation (NMOR). The system uses adaptive experimental design to iteratively select optimal bias fields that maximize information gain about unknown magnetic fields.

## Project Structure

```
bayesian_nmor/
├── BayesianNMOR/              # Core Bayesian inference implementation
│   ├── bayes_experiment.py    # Main experimental utilities and algorithms
│   ├── toy_bayesian_model.ipynb  # Interactive notebook with examples
│   └── tests/                 # Unit tests and test documentation
├── FPGA/                      # Hardware control and data acquisition
│   ├── fpga.py               # Red Pitaya FPGA control via SCPI
│   └── scpi_loadtest.py      # SCPI communication performance testing
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Key Features

### Bayesian Inference Engine (`BayesianNMOR/`)

- **Adaptive Experimental Design**: Automatically selects optimal bias fields to maximize information gain
- **Dispersive Lorentzian Signal Model**: Implements the theoretical model for NMOR signals
- **Real-time Posterior Updates**: Bayesian updating with Gaussian measurement noise
- **Information-theoretic Optimization**: Uses KL divergence to quantify and maximize information gain

### Hardware Integration (`FPGA/`)

- **Real-time Control**: Direct communication with Red Pitaya FPGA boards via SCPI protocol
- **PID Controller Interface**: Configure and control feedback loops for magnetic field stabilization  
- **Data Acquisition**: Oscilloscope functionality for capturing measurement traces
- **Performance Monitoring**: Latency testing and system optimization tools

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/apranav22/bayesian_nmor.git
cd bayesian_nmor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run tests to verify installation:**
```bash
pytest -q
```

### Basic Usage

#### Running Bayesian Experiments

```python
from BayesianNMOR.bayes_experiment import run_experiment

# Run an adaptive Bayesian experiment
B_grid, posteriors, biases = run_experiment(
    N_exp=5,           # Number of experimental iterations
    B_true=0.2,        # True unknown field (for simulation)
    Gamma=1.0,         # Lorentzian width parameter
    sigma=0.05,        # Measurement noise standard deviation
    B_range=(-1.0, 1.0), # Field search range
    n_grid=500         # Grid resolution for posterior
)

print("Optimal bias fields chosen:", biases)
```

#### Hardware Control

```python
from FPGA.fpga import set_pid, set_setpoint, capture_trace

# Configure PID controller
set_pid(kp=0.3, ki=1.0, kd=0.0)

# Set field setpoint
set_setpoint(0.2)  # 0.2 V setpoint

# Capture measurement trace
capture_trace("measurement.csv", dec=64)
```

#### Interactive Exploration

Open the Jupyter notebook for interactive examples:
```bash
jupyter notebook BayesianNMOR/toy_bayesian_model.ipynb
```

## Core Algorithms

### Signal Model
The system models NMOR signals using a dispersive Lorentzian:

```
α(B) = B_total / (B_total² + Γ²)
```

where `B_total = B_unknown + B_bias` and `Γ` is the resonance width.

### Adaptive Design
The algorithm selects bias fields `B_bias` that maximize expected information gain:

1. **Prior Update**: Start with uniform prior over field range
2. **Utility Calculation**: Compute expected KL divergence for candidate bias fields
3. **Optimal Selection**: Choose bias that maximizes expected information gain
4. **Measurement**: Acquire noisy measurements at selected bias
5. **Posterior Update**: Apply Bayes' rule to update field estimate
6. **Iteration**: Repeat until convergence or iteration limit

## Hardware Requirements

- **Red Pitaya FPGA Board**: For real-time control and data acquisition
- **Network Connection**: Ethernet connection to FPGA board
- **Magnetic Field Generation**: Coils or other field sources controlled via PID

## Testing

The project includes comprehensive unit tests covering:

- Signal model properties (odd function, boundedness)
- Likelihood function correctness
- Posterior normalization and convergence
- Information-theoretic measures
- End-to-end experimental workflows

Run the full test suite:
```bash
pytest tests/ -v
```

For testing documentation, see [`BayesianNMOR/tests/TESTING.md`](BayesianNMOR/tests/TESTING.md).

## License

[Add appropriate license information]

## Citation

## Contact

**Author**: Dinesh R, Pranav Agrawal
**Repository**: https://github.com/apranav22/bayesian_nmor

For questions or collaboration inquiries, please open an issue on GitHub.
# Lumerical Python API

## Overview

The Lumerical Python API provides a comprehensive interface for controlling and automating ANSYS Lumerical's photonic simulation tools through Python. This API enables users to programmatically create, modify, run simulations, and analyze results from Lumerical's suite of products including FDTD Solutions, MODE Solutions, DEVICE, and INTERCONNECT.

## Key Features

- **Complete Control**: Programmatically control all aspects of Lumerical's simulation tools
- **Automation**: Create, modify, run, and analyze simulations without manual intervention
- **Data Analysis**: Extract and process simulation results directly in Python
- **Optimization**: Perform advanced optimization of photonic designs using the `lumopt` module
- **Cloud Computing**: Run simulations on AWS cloud infrastructure using the `lumerical.aws` module
- **Distributed Computing**: Utilize SLURM-based cluster computing with the `lumslurm` module

## Installation

The Lumerical Python API is automatically installed with ANSYS Lumerical v242. The API is located in:

```
ANSYS_INSTALL_DIR/Lumerical/v242/api/python
```

To use the API, you need to ensure this directory is in your Python path:

```python
import sys
sys.path.append("ANSYS_INSTALL_DIR/Lumerical/v242/api/python")
import lumapi
```

## Requirements

- Python 3.x
- ANSYS Lumerical v242 or later
- Required Python packages: numpy, scipy (for optimization)

## Basic Usage

```python
import lumapi

# Open a FDTD Solutions session
fdtd = lumapi.FDTD()

# Create a simple waveguide simulation
fdtd.addfdtd(dimension="2D", x=0, y=0, x_span=10e-6, y_span=6e-6)
fdtd.addwaveguide(x=0, y=0, z_span=0.22e-6, material="Si")
fdtd.addmode(direction="Forward", injection_axis="x", x=-4e-6)

# Run the simulation
fdtd.run()

# Get results
T = fdtd.getresult("monitor", "T")

# Close the session
fdtd.close()
```

## Modules

The API consists of several modules:

- **lumapi**: Core module for controlling Lumerical products
- **lumopt**: Module for photonic optimization
- **lumerical.aws**: Module for running simulations on AWS
- **lumslurm**: Module for distributed computing using SLURM

## Documentation

For detailed documentation on functions and classes, please refer to the User Manual.

## License

This software is commercial software and can be used under the terms of the Ansys License Agreement as published by ANSYS, Inc.

## Support

For support, please contact ANSYS Lumerical support at [support@lumerical.com](mailto:support@lumerical.com) or visit [www.ansys.com/support](https://www.ansys.com/support).

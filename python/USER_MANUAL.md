# Lumerical Python API - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Importing the API](#importing-the-api)
3. [Core API (lumapi)](#core-api-lumapi)
   - [Opening and Closing Sessions](#opening-and-closing-sessions)
   - [Working with Files](#working-with-files)
   - [Running Simulations](#running-simulations)
   - [Managing Objects](#managing-objects)
   - [Getting and Setting Variables](#getting-and-setting-variables)
   - [Accessing Results](#accessing-results)
   - [Scripting Functions](#scripting-functions)
4. [Optimization (lumopt)](#optimization-lumopt)
   - [Optimization Framework](#optimization-framework)
   - [Figures of Merit](#figures-of-merit)
   - [Geometries](#geometries)
   - [Optimizers](#optimizers)
5. [Cloud Computing (lumerical.aws)](#cloud-computing-lumericalaws)
   - [Setting Up AWS Resources](#setting-up-aws-resources)
   - [Running Simulations on AWS](#running-simulations-on-aws)
   - [Managing AWS Resources](#managing-aws-resources)
6. [Distributed Computing (lumslurm)](#distributed-computing-lumslurm)
   - [Running Simulations with SLURM](#running-simulations-with-slurm)
   - [Parameter Sweeps with SLURM](#parameter-sweeps-with-slurm)
7. [Advanced Topics](#advanced-topics)
   - [Error Handling](#error-handling)
   - [Performance Optimization](#performance-optimization)
   - [Debugging Tips](#debugging-tips)
8. [Examples](#examples)

## Introduction

The Lumerical Python API provides a comprehensive interface for controlling and automating ANSYS Lumerical's photonic simulation tools through Python. This API enables users to programmatically create, modify, run simulations, and analyze results from Lumerical's suite of products including FDTD Solutions, MODE Solutions, DEVICE, and INTERCONNECT.

This user manual provides detailed information on how to use the API effectively, covering all available functions and modules.

## Getting Started

### Installation

The Lumerical Python API is automatically installed with ANSYS Lumerical v242. The API is located in:

```
ANSYS_INSTALL_DIR/Lumerical/v242/api/python
```

### Importing the API

To use the API, you need to ensure the API directory is in your Python path:

```python
import sys
sys.path.append("ANSYS_INSTALL_DIR/Lumerical/v242/api/python")
import lumapi
```

## Core API (lumapi)

The `lumapi` module is the core module for controlling Lumerical products.

### Opening and Closing Sessions

#### Opening a Session

You can open a session with any of Lumerical's products using the following classes:

```python
# Open FDTD Solutions
fdtd = lumapi.FDTD(filename=None, hide=False)

# Open MODE Solutions
mode = lumapi.MODE(filename=None, hide=False)

# Open DEVICE
device = lumapi.DEVICE(filename=None, hide=False)

# Open INTERCONNECT
interconnect = lumapi.INTERCONNECT(filename=None, hide=False)
```

Parameters:
- `filename` (optional): Path to a Lumerical file to open (.fsp, .lms, .ldev, .icp)
- `hide` (optional): Boolean to hide the GUI (default: False)
- `key` (optional): License key
- `serverArgs` (optional): Dictionary of server arguments
- `remoteArgs` (optional): Dictionary for remote connections

Alternatively, you can use the generic `open` function:

```python
handle = lumapi.open("fdtd", filename=None, hide=False)
```

#### Closing a Session

Always close your sessions when done:

```python
fdtd.close()
# or
lumapi.close(handle)
```

You can also use the context manager pattern:

```python
with lumapi.FDTD() as fdtd:
    # Your code here
    # Session will automatically close when exiting the block
```

### Working with Files

#### Loading and Saving Files

```python
# Load a file
fdtd.load("simulation.fsp")

# Save a file
fdtd.save("simulation.fsp")
```

#### Running Script Files

```python
# Run a Lumerical script file (.lsf)
fdtd.feval("script.lsf")

# Evaluate a Lumerical script command
fdtd.eval("addfdtd;")
```

### Running Simulations

```python
# Run the current simulation
fdtd.run()

# Run a specific analysis
fdtd.run("analysis1")
```

### Managing Objects

#### Adding Objects

Lumerical objects can be added using various `add*` functions:

```python
# Add an FDTD simulation region
fdtd.addfdtd(x=0, y=0, z=0, x_span=2e-6, y_span=2e-6, z_span=2e-6)

# Add a rectangle
fdtd.addrect(x=0, y=0, z=0, x_span=1e-6, y_span=1e-6, z_span=0.22e-6, material="Si")

# Add a mode source
fdtd.addmode(name="source", injection_axis="x", direction="forward", x=-1e-6)

# Add a monitor
fdtd.addpower(name="monitor", monitor_type="2D X-normal", x=1e-6)
```

#### Accessing and Modifying Objects

Objects can be accessed and modified using their names:

```python
# Get object properties
material = fdtd.getnamed("rectangle", "material")

# Set object properties
fdtd.setnamed("rectangle", "material", "SiO2")

# Delete an object
fdtd.delete("rectangle")
```

### Getting and Setting Variables

```python
# Get a variable
value = fdtd.getv("variable_name")
# or
value = fdtd.getvar("variable_name")

# Set a variable
fdtd.putv("variable_name", value)
# or
fdtd.setvar("variable_name", value)
```

### Accessing Results

```python
# Get a result from a monitor
result = fdtd.getresult("monitor_name", "field")

# Access specific data from the result
E = result["E"]
x = result["x"]
```

### Scripting Functions

The API supports all Lumerical script functions. Here are some common ones:

```python
# Set the simulation mesh
fdtd.setmesh(dx=10e-9, dy=10e-9, dz=10e-9)

# Set the simulation time
fdtd.settime(100e-15)

# Get the number of objects with a specific name
count = fdtd.getnamednumber("rectangle")

# Select objects
fdtd.select("rectangle")

# Copy objects
fdtd.copy()

# Paste objects
fdtd.paste()
```

## Optimization (lumopt)

The `lumopt` module provides tools for optimizing photonic designs.

### Optimization Framework

```python
from lumopt.optimization import Optimization
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers

# Define the geometry to optimize
geometry = FunctionDefinedPolygon(...)

# Define the figure of merit
fom = ModeMatch(...)

# Set up the optimization
opt = Optimization(base_script="base_sim.lsf",
                  wavelengths=[1550e-9],
                  fom=fom,
                  geometry=geometry,
                  optimizer=ScipyOptimizers(...)
                  )

# Run the optimization
opt.run()
```

### Figures of Merit

The `lumopt.figures_of_merit` module provides different figures of merit for optimization:

- `ModeMatch`: Optimize for mode matching
- `DiffractionEfficiency`: Optimize for diffraction efficiency
- `WaveguideCoupling`: Optimize for waveguide coupling efficiency

### Geometries

The `lumopt.geometries` module provides different geometry parameterizations:

- `FunctionDefinedPolygon`: Define polygon vertices using functions
- `PolygonSet`: Define a set of polygons
- `TopologyOptimization`: Perform topology optimization

### Optimizers

The `lumopt.optimizers` module provides different optimization algorithms:

- `ScipyOptimizers`: Wrapper for SciPy optimization algorithms
- `FixedStepGradientDescent`: Simple gradient descent with fixed step size
- `AdaptiveGradientDescent`: Gradient descent with adaptive step size

## Cloud Computing (lumerical.aws)

The `lumerical.aws` module provides tools for running simulations on AWS.

### Setting Up AWS Resources

```python
import lumerical.aws as aws

# Create a virtual private cloud (VPC)
aws.create_virtual_private_cloud("my-vpc")

# Initialize the VPC with a license file
aws.initialize_virtual_private_cloud("my-vpc", "license.lic", "ami-id", "lic-ami-id")
```

### Running Simulations on AWS

```python
# Start the license server
aws.start_license_server("my-vpc")

# Configure compute instances
aws.configure_compute_instance("my-vpc", instance_type="c5.4xlarge")

# Start compute instances
workgroup_id = aws.start_compute_instances("my-vpc", num_instances=4)

# Run a parameter sweep
aws.run_parameter_sweep("my-vpc", workgroup_id, "s3://bucket/project.fsp")
```

### Managing AWS Resources

```python
# Stop a job
aws.stop_job("my-vpc", workgroup_id)

# Terminate all instances
aws.terminate_all_instances("my-vpc")

# Remove the VPC
aws.remove_virtual_private_cloud("my-vpc")
```

## Distributed Computing (lumslurm)

The `lumslurm` module provides tools for running simulations on SLURM clusters.

### Running Simulations with SLURM

```python
import lumslurm

# Run a simulation
job_id = lumslurm.run_solve("simulation.fsp", 
                            partition="compute", 
                            nodes=2, 
                            processes_per_node=8)

# Run a script after the simulation
script_id = lumslurm.run_script(script_file="process.py", 
                               fsp_file="simulation.fsp", 
                               dependency=job_id)
```

### Parameter Sweeps with SLURM

```python
# Run a parameter sweep
sweep_id = lumslurm.run_sweep("simulation.fsp", 
                             sweep_name="parameter_sweep", 
                             solve_partition="compute", 
                             solve_nodes=2)

# Run a batch of simulations
batch_id = lumslurm.run_batch("simulations/*.fsp", 
                             postprocess_script="process.py")
```

## Advanced Topics

### Error Handling

```python
try:
    fdtd.run()
except lumapi.LumApiError as e:
    print(f"Simulation error: {e}")
```

### Performance Optimization

- Use `hide=True` when opening sessions to avoid GUI overhead
- Use distributed computing for large simulations
- Consider using cloud resources for massive parameter sweeps

### Debugging Tips

- Use `fdtd.eval("?")` to see available commands
- Use `print(dir(fdtd))` to see available methods
- Check simulation log files for errors

## Examples

### Basic FDTD Simulation

```python
import lumapi

with lumapi.FDTD() as fdtd:
    # Create a simple waveguide simulation
    fdtd.addfdtd(dimension="2D", x=0, y=0, x_span=10e-6, y_span=6e-6)
    fdtd.addwaveguide(x=0, y=0, z_span=0.22e-6, material="Si")
    fdtd.addmode(direction="Forward", injection_axis="x", x=-4e-6)
    
    # Run the simulation
    fdtd.run()
    
    # Get results
    T = fdtd.getresult("monitor", "T")
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(T["lambda"], T["T"])
    plt.xlabel("Wavelength (m)")
    plt.ylabel("Transmission")
    plt.show()
```

### Optimization Example

```python
import lumapi
import numpy as np
from lumopt.optimization import Optimization
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers

# Define the geometry to optimize
def get_vertices(params):
    # Convert parameters to polygon vertices
    vertices = []
    # ... calculation logic here
    return vertices

geometry = FunctionDefinedPolygon(func=get_vertices, initial_params=np.array([1, 2, 3]))

# Define the figure of merit
fom = ModeMatch(monitor_name="monitor", mode_number=1, direction="Forward")

# Set up the optimization
opt = Optimization(base_script="base_sim.lsf",
                  wavelengths=[1550e-9],
                  fom=fom,
                  geometry=geometry,
                  optimizer=ScipyOptimizers(method="L-BFGS-B", scaling_factor=1e6)
                  )

# Run the optimization
results = opt.run()
```

### SLURM Parameter Sweep

```python
import lumslurm

# Run a parameter sweep on a SLURM cluster
sweep_id = lumslurm.run_sweep("grating_coupler.fsp", 
                             sweep_name="angle_sweep", 
                             solve_partition="compute", 
                             solve_nodes=2,
                             solve_processes_per_node=8,
                             solve_threads_per_process=2,
                             script_partition="postprocess",
                             script_threads=4,
                             block=True)
```

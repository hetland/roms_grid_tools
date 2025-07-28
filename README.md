# ROMS Grid Tools

A Python package for working with ROMS (Regional Ocean Modeling System) nested grids. This package provides tools for generating contact points between nested grids, computing grid metrics, and handling grid interpolation weights.

This is a Python translation (with AI assist) of the MATLAB-based grid tools found [here](https://github.com/DOI-USGS/COAWST/tree/main/Tools/mfiles/rutgers/grid), in particular the tools used for creating nested grids.

## Features

- **Contact Point Generation**: Create contact points between nested ROMS grids
- **Grid Utilities**: Read ROMS grid files and compute grid metrics
- **Interpolation Weights**: Calculate linear and quadratic interpolation weights
- **Nested Grid Support**: Handle grid refinement and coincident grid connections
- **NetCDF I/O**: Read and write ROMS-compatible NetCDF files
- **Visualization**: Plot grid perimeters and contact points

## Installation

### From PyPI (when available)
```bash
pip install roms-grid-tools
```

### From Source
```bash
git clone https://github.com/hetland/roms_grid_tools.git
cd roms_grid_tools
pip install -e .
```

### Dependencies
The package requires the following core dependencies:
- numpy >= 1.20.0
- scipy >= 1.7.0
- netCDF4 >= 1.5.0
- matplotlib >= 3.3.0
- xarray >= 0.19.0

See `requirements.txt` for the complete list of dependencies.

## Quick Start

```python
import roms_grid_tools as rgt

# Define your grid files (in order of nesting)
grid_files = [
    'parent_grid.nc',
    'child_grid_1.nc',
    'child_grid_2.nc'
]

# Generate contact points
S, G = rgt.contact(
    gnames=grid_files,
    cname='contact_points.nc',
    lmask=True,        # Remove land contact points
    mask_interp=False, # Compute masks from RHO-mask
    lplot=True         # Create plots
)

print(f"Generated contact points for {S.Ngrids} grids")
print(f"Total contact points: {S.Ndatum}")
```

## Package Structure

```
roms_grid_tools/
├── __init__.py              # Main package imports
├── contact.py               # Main contact points function
├── data_structures.py       # Data classes and structures
├── grid_utils.py           # General grid utilities
├── nested_grids.py         # Nested grid specific functions
└── weights.py              # Interpolation weight calculations
```

### Core Modules

#### `contact.py`
Main function for generating contact points between nested grids.

#### `grid_utils.py`
General-purpose grid utilities:
- `get_roms_grid()`: Read ROMS grid NetCDF files
- `grid_metrics()`: Compute grid metrics (pm, pn, dndx, dmde)
- `gcircle()`: Great circle distance calculations
- `inpolygon_numpy()`: Point-in-polygon testing

#### `weights.py`
Interpolation weight calculations:
- `linear_weights()`: 4-point bilinear interpolation weights
- `quadratic_weights()`: 9-point biquadratic interpolation weights
- `validate_weights()`: Weight validation utilities

#### `nested_grids.py`
Nested grid specific functionality:
- `grid_perimeter()`: Set grid perimeters and boundary edges
- `grid_connections()`: Determine grid connectivity
- `boundary_contact()`: Identify boundary contact points
- `write_contact()`: Write contact points to NetCDF
- `plot_contact()`: Visualize contact regions

#### `data_structures.py`
Data classes for organizing grid and contact point information:
- `ContactStructure`: Main structure for nested grid information
- `ContactPoint`: Contact point data between grids
- `ContactRegion`: Information about grid connections
- `GridInfo`: Grid dimension and metadata

## Examples

### Basic Usage
```python
import roms_grid_tools as rgt

# Simple contact point generation
grid_files = ['coarse.nc', 'fine.nc']
S, G = rgt.contact(grid_files, 'contact.nc')
```

### Grid Utilities
```python
from roms_grid_tools import get_roms_grid, grid_metrics

# Read a grid file
grid = get_roms_grid('my_grid.nc')
print(f"Grid dimensions: {grid['Lp']} x {grid['Mp']}")

# Compute grid metrics
pm, pn, dndx, dmde = grid_metrics(grid)
print(f"Grid spacing range: {1/pm.max():.1f} - {1/pm.min():.1f} m")
```

### Interpolation Weights
```python
from roms_grid_tools import linear_weights, quadratic_weights
import numpy as np

# Example contact points
xrg = np.array([1.5, 2.3, 3.7])  # Receiver grid XI coordinates
erg = np.array([2.1, 3.4, 1.9])  # Receiver grid ETA coordinates
idg = np.array([1, 2, 3])        # Donor grid I indices  
jdg = np.array([2, 3, 1])        # Donor grid J indices

# Compute interpolation weights
linear_w = linear_weights(xrg, erg, idg, jdg)
quad_w = quadratic_weights(xrg, erg, idg, jdg)

print(f"Linear weights shape: {linear_w.shape}")    # (4, 3)
print(f"Quadratic weights shape: {quad_w.shape}")   # (9, 3)
```

## Development

### Setting up Development Environment
```bash
git clone https://github.com/hetland/roms_grid_tools.git
cd roms_grid_tools
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black roms_grid_tools/
flake8 roms_grid_tools/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original MATLAB implementation from the COAWST project
- ROMS/TOMS community for the original algorithms
- Contributors to the Python translation

## Related Projects

- [ROMS](https://www.myroms.org/): Regional Ocean Modeling System
- [COAWST](https://github.com/DOI-USGS/COAWST): Coupled Ocean-Atmosphere-Wave-Sediment Transport modeling system
- [pyroms](https://github.com/ESMG/pyroms): Python tools for ROMS

## Citation

If you use this software in your research, please cite:

```
ROMS Grid Tools: Python tools for ROMS nested grids (2025)
https://github.com/hetland/roms_grid_tools
```
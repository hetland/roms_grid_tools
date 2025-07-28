"""
ROMS Grid Tools

A Python package for working with ROMS (Regional Ocean Modeling System) nested grids.
This package provides tools for generating contact points between nested grids,
computing grid metrics, and handling grid interpolation weights.

This is a Python translation of MATLAB-based grid tools found in the COAWST project.
"""

__version__ = "0.1.0"
__author__ = "ROMS Grid Tools Contributors"

# Import main functions for easy access
try:
    from .contact import contact
    from .data_structures import (
        ContactStructure,
        ContactPoint,
        ContactRegion,
        GridInfo
    )
    from .grid_utils import (
        get_roms_grid,
        grid_metrics,
        inpolygon_numpy,
        grids_structure,
        gcircle
    )
    from .weights import (
        linear_weights,
        quadratic_weights,
        validate_weights
    )
    from .nested_grids import (
        grid_perimeter,
        grid_connections,
        boundary_contact,
        write_contact,
        plot_contact
    )

    __all__ = [
        # Main function
        "contact",
        # Data structures
        "ContactStructure", 
        "ContactPoint",
        "ContactRegion",
        "GridInfo",
        # Grid utilities
        "get_roms_grid",
        "grid_metrics",
        "inpolygon_numpy",
        "grids_structure",
        "gcircle",
        # Interpolation weights
        "linear_weights",
        "quadratic_weights",
        "validate_weights",
        # Nested grid functions
        "grid_perimeter",
        "grid_connections", 
        "boundary_contact",
        "write_contact",
        "plot_contact",
    ]

except ImportError as e:
    # Handle import errors gracefully
    import warnings
    warnings.warn(f"Some imports failed: {e}. Check dependencies.")
    
    # Minimal exports for when imports fail
    __all__ = []

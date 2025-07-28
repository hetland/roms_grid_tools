#!/usr/bin/env python3
"""
Example usage of the ROMS contact points generator.

This script demonstrates how to use the Python version of the 
MATLAB contact.m function for ROMS nested grids.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from contact import contact, ContactStructure, ContactPoint
    print("✓ Successfully imported contact module")
except ImportError as e:
    print(f"✗ Failed to import contact module: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def example_basic_usage():
    """
    Basic example of using the contact function.
    """
    print("\n=== Basic Usage Example ===")
    
    # Example grid files (these would be actual NetCDF files)
    grid_files = [
        'parent_grid.nc',     # Coarse parent grid
        'child_grid_01.nc',   # First nested grid
        'child_grid_02.nc'    # Second nested grid
    ]
    
    # Output contact points file
    contact_file = 'contact_points.nc'
    
    try:
        # Generate contact points
        print(f"Processing grids: {grid_files}")
        print(f"Output file: {contact_file}")
        
        # Note: This will raise NotImplementedError since dependency functions
        # are not yet implemented
        S, G = contact(
            gnames=grid_files,
            cname=contact_file,
            lmask=False,        # Keep land contact points
            mask_interp=False,  # Compute masks from RHO-mask  
            lplot=False         # No plotting
        )
        
        print(f"✓ Successfully generated contact points")
        print(f"  Number of grids: {S.Ngrids}")
        print(f"  Number of contact regions: {S.Ncontact}")
        print(f"  Total contact points: {S.Ndatum}")
        
    except NotImplementedError as e:
        print(f"⚠ Function not fully implemented: {e}")
        print("This is expected - dependency functions need to be implemented first.")


def example_weight_calculation():
    """
    Example of calculating interpolation weights.
    """
    print("\n=== Interpolation Weights Example ===")
    
    try:
        from contact import linear_weights, quadratic_weights
        
        # Example contact point data
        npoints = 5
        xrg = np.array([1.5, 2.3, 3.7, 4.1, 5.8])  # Receiver grid XI coords
        erg = np.array([2.1, 3.4, 1.9, 4.2, 2.7])  # Receiver grid ETA coords
        idg = np.array([1, 2, 3, 4, 5])             # Donor grid I indices
        jdg = np.array([2, 3, 1, 4, 2])             # Donor grid J indices
        
        print(f"Calculating weights for {npoints} contact points")
        
        # Calculate linear weights
        linear_w = linear_weights(xrg, erg, idg, jdg)
        print(f"Linear weights shape: {linear_w.shape}")
        print(f"Linear weights sum: {np.sum(linear_w, axis=0)}")
        
        # Calculate quadratic weights  
        quad_w = quadratic_weights(xrg, erg, idg, jdg)
        print(f"Quadratic weights shape: {quad_w.shape}")
        print(f"Quadratic weights sum: {np.sum(quad_w, axis=0)}")
        
        # Check that weights sum to unity (or close to it)
        linear_sums = np.sum(linear_w, axis=0)
        quad_sums = np.sum(quad_w, axis=0)
        
        print(f"Linear weights unity check: {np.allclose(linear_sums, 1.0)}")
        print(f"Quadratic weights unity check: {np.allclose(quad_sums, 1.0)}")
        
    except Exception as e:
        print(f"✗ Error in weight calculation: {e}")


def example_data_structures():
    """
    Example of working with contact data structures.
    """
    print("\n=== Data Structures Example ===")
    
    # Create example contact structure
    S = ContactStructure()
    S.Ngrids = 2
    S.Ncontact = 2
    S.spherical = True
    
    # Create example contact points
    contact_point = ContactPoint()
    contact_point.Irg_rho = np.array([10, 11, 12, 13, 14])
    contact_point.Jrg_rho = np.array([20, 21, 22, 23, 24])
    contact_point.h = np.array([100.5, 95.2, 88.7, 92.1, 87.3])
    contact_point.mask_rho = np.ones(5)  # All water points
    
    print(f"Contact structure created:")
    print(f"  Number of grids: {S.Ngrids}")
    print(f"  Number of contact regions: {S.Ncontact}")
    print(f"  Spherical coordinates: {S.spherical}")
    print(f"  Contact points: {len(contact_point.Irg_rho)}")
    print(f"  Bathymetry range: {np.min(contact_point.h):.1f} - {np.max(contact_point.h):.1f} m")


def main():
    """
    Main function to run all examples.
    """
    print("ROMS Contact Points Generator - Python Examples")
    print("=" * 50)
    
    # Run examples
    example_data_structures()
    example_weight_calculation()
    example_basic_usage()
    
    print("\n=== Summary ===")
    print("This example demonstrates the Python translation structure.")
    print("To use with real data, implement the dependency functions:")
    print("  - grids_structure(), grid_perimeter(), grid_connections()")
    print("  - boundary_contact(), write_contact(), plot_contact()")
    print("  - refined_gridvar(), roms_metrics(), uvp_masks()")
    print("\nSee README.md for detailed implementation guide.")


if __name__ == "__main__":
    main()

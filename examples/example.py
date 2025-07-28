#!/usr/bin/env python3
"""
Example usage of the ROMS contact points generator.

This script demonstrates how to use the Python version of the 
MATLAB contact.m function for ROMS nested grids.
"""

import numpy as np
import sys
import os

# Add the parent directory to Python path so we can import roms_grid_tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import roms_grid_tools as rgt
    print("✓ Successfully imported roms_grid_tools")
    print(f"  Version: {rgt.__version__}")
except ImportError as e:
    print(f"✗ Failed to import roms_grid_tools: {e}")
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
        
        # Note: This will raise errors since grid files don't exist
        # In real usage, these would be actual ROMS grid NetCDF files
        S, G = rgt.contact(
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
        
    except Exception as e:
        print(f"⚠ Function failed (expected with example files): {e}")
        print("This is expected since the example grid files don't exist.")
        print("In real usage, provide actual ROMS grid NetCDF files.")


def example_weight_calculation():
    """
    Example of calculating interpolation weights.
    """
    print("\n=== Interpolation Weights Example ===")
    
    try:
        # Example contact point data
        npoints = 5
        xrg = np.array([1.5, 2.3, 3.7, 4.1, 5.8])  # Receiver grid XI coords
        erg = np.array([2.1, 3.4, 1.9, 4.2, 2.7])  # Receiver grid ETA coords
        idg = np.array([1, 2, 3, 4, 5])             # Donor grid I indices
        jdg = np.array([2, 3, 1, 4, 2])             # Donor grid J indices
        
        print(f"Calculating weights for {npoints} contact points")
        
        # Calculate linear weights
        linear_w = rgt.linear_weights(xrg, erg, idg, jdg)
        print(f"Linear weights shape: {linear_w.shape}")
        print(f"Linear weights sum: {np.sum(linear_w, axis=0)}")
        
        # Calculate quadratic weights  
        quad_w = rgt.quadratic_weights(xrg, erg, idg, jdg)
        print(f"Quadratic weights shape: {quad_w.shape}")
        print(f"Quadratic weights sum: {np.sum(quad_w, axis=0)}")
        
        # Check that weights sum to unity (or close to it)
        linear_sums = np.sum(linear_w, axis=0)
        quad_sums = np.sum(quad_w, axis=0)
        
        print(f"Linear weights unity check: {np.allclose(linear_sums, 1.0)}")
        print(f"Quadratic weights unity check: {np.allclose(quad_sums, 1.0)}")
        
        # Validate weights
        linear_valid = rgt.validate_weights(linear_w)
        quad_valid = rgt.validate_weights(quad_w)
        print(f"Linear weights valid: {linear_valid}")
        print(f"Quadratic weights valid: {quad_valid}")
        
    except Exception as e:
        print(f"✗ Error in weight calculation: {e}")


def example_data_structures():
    """
    Example of working with contact data structures.
    """
    print("\n=== Data Structures Example ===")
    
    # Create example contact structure
    S = rgt.ContactStructure()
    S.Ngrids = 2
    S.Ncontact = 2
    S.spherical = True
    
    # Create example contact points
    contact_point = rgt.ContactPoint()
    contact_point.Irg_rho = np.array([10, 11, 12, 13, 14])
    contact_point.Jrg_rho = np.array([20, 21, 22, 23, 24])
    contact_point.h = np.array([100.5, 95.2, 88.7, 92.1, 87.3])
    contact_point.mask_rho = np.ones(5)  # All water points
    
    print(f"Contact structure created:")
    print(f"  Number of grids: {S.Ngrids}")
    print(f"  Number of contact regions: {S.Ncontact}")
    print(f"  Spherical coordinates: {S.spherical}")
    print(f"  Contact points: {contact_point.get_npoints('rho')}")
    print(f"  Bathymetry range: {np.min(contact_point.h):.1f} - {np.max(contact_point.h):.1f} m")
    
    # Print summary
    S.print_summary()


def example_grid_utilities():
    """
    Example of grid utility functions.
    """
    print("\n=== Grid Utilities Example ===")
    
    try:
        # Great circle distance calculation
        dist = rgt.gcircle(-122.0, 37.0, -121.0, 38.0)
        print(f"Great circle distance: {dist:.1f} km")
        
        # Point in polygon test
        # Define a simple square polygon
        xv = np.array([0, 1, 1, 0, 0])
        yv = np.array([0, 0, 1, 1, 0])
        
        # Test points
        xq = np.array([0.5, 1.5, 0.5])
        yq = np.array([0.5, 0.5, 1.5])
        
        inside = rgt.inpolygon_numpy(xq, yq, xv, yv)
        print(f"Points inside polygon: {inside}")  # [True, False, False]
        
        for i, (x, y, is_in) in enumerate(zip(xq, yq, inside)):
            status = "inside" if is_in else "outside"
            print(f"  Point {i+1}: ({x}, {y}) is {status}")
            
    except Exception as e:
        print(f"✗ Error in grid utilities: {e}")


def main():
    """
    Main function to run all examples.
    """
    print("ROMS Contact Points Generator - Python Examples")
    print("=" * 50)
    
    # Run examples
    example_data_structures()
    example_weight_calculation()
    example_grid_utilities()
    example_basic_usage()
    
    print("\n=== Summary ===")
    print("This example demonstrates the Python translation structure.")
    print("To use with real data:")
    print("  1. Create or obtain ROMS grid NetCDF files")
    print("  2. Use rgt.contact() with actual grid file names")
    print("  3. The package will generate contact points and weights")
    print("  4. Output will be a NetCDF file with contact point data")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()

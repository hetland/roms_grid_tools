"""
Grid utilities for ROMS nested grids.

This module contains general-purpose utilities for working with ROMS grid files,
including reading grid data, computing metrics, and basic geometric operations.
"""

import numpy as np
import netCDF4
from typing import Dict, Any, Optional, Tuple
import warnings


def get_roms_grid(filename: str) -> Dict[str, Any]:
    """
    Read ROMS grid NetCDF file and extract grid variables.
    
    This function reads a ROMS grid NetCDF file and extracts all relevant
    grid variables including coordinates, dimensions, metrics, and masks.
    
    Parameters:
    -----------
    filename : str
        Path to ROMS grid NetCDF file
        
    Returns:
    --------
    grid_dict : Dict[str, Any]
        Dictionary containing all grid variables
        
    Examples:
    ---------
    >>> grid = get_roms_grid('my_grid.nc')
    >>> print(f"Grid dimensions: {grid['Lp']} x {grid['Mp']}")
    >>> print(f"Spherical: {grid['spherical']}")
    """
    try:
        with netCDF4.Dataset(filename, 'r') as ncfile:
            # Read basic grid dimensions and variables
            grid_dict = {
                'grid_name': filename,
                'spherical': bool(ncfile.variables.get('spherical', [1])[0] if 'spherical' in ncfile.variables else True),
            }
            
            # Read dimensions
            if 'xi_rho' in ncfile.dimensions:
                grid_dict['Lp'] = len(ncfile.dimensions['xi_rho'])
                grid_dict['L'] = grid_dict['Lp'] - 1
                grid_dict['Lm'] = grid_dict['L'] - 1
            
            if 'eta_rho' in ncfile.dimensions:
                grid_dict['Mp'] = len(ncfile.dimensions['eta_rho'])
                grid_dict['M'] = grid_dict['Mp'] - 1
                grid_dict['Mm'] = grid_dict['M'] - 1
                
            # Read coordinate variables if available
            coord_vars = ['lon_rho', 'lat_rho', 'lon_psi', 'lat_psi',
                         'lon_u', 'lat_u', 'lon_v', 'lat_v',
                         'x_rho', 'y_rho', 'x_psi', 'y_psi',
                         'x_u', 'y_u', 'x_v', 'y_v']
            
            for var in coord_vars:
                if var in ncfile.variables:
                    grid_dict[var] = ncfile.variables[var][:]
                    
            # Read other grid variables
            grid_vars = ['h', 'f', 'pm', 'pn', 'angle', 'mask_rho', 'mask_psi', 'mask_u', 'mask_v']
            for var in grid_vars:
                if var in ncfile.variables:
                    grid_dict[var] = ncfile.variables[var][:]
                    
            # Read nested grid parameters if present
            nested_vars = ['refine_factor', 'parent_grid', 'parent_Imin', 'parent_Imax', 
                          'parent_Jmin', 'parent_Jmax']
            for var in nested_vars:
                if var in ncfile.variables:
                    grid_dict[var] = ncfile.variables[var][:]
                elif hasattr(ncfile, var):
                    grid_dict[var] = getattr(ncfile, var)
                    
            return grid_dict
            
    except Exception as e:
        warnings.warn(f"Could not read {filename}: {e}. Returning basic example grid structure.")
        # Return basic placeholder structure
        return {
            'grid_name': filename,
            'spherical': False,
            'Lm': 100, 'Mm': 80,
            'Lp': 101, 'Mp': 81,
            'L': 100, 'M': 80
        }


def grid_metrics(G: Dict[str, Any], great_circle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROMS Grid horizontal metrics.
    
    This function computes the curvilinear coordinate metrics (pm, pn) and their
    derivatives (dndx, dmde) for a ROMS grid. These metrics are used to convert
    between curvilinear and physical coordinates.
    
    Parameters:
    -----------
    G : Dict[str, Any]
        Grid structure dictionary or NetCDF filename
    great_circle : bool, optional
        Switch indicating how to compute grid distance (default True)
        If True and spherical=True, uses great circle distances
        If False, uses Cartesian distances
        
    Returns:
    --------
    pm : np.ndarray
        Curvilinear coordinate metric in XI-direction (1/meters)
    pn : np.ndarray  
        Curvilinear coordinate metric in ETA-direction (1/meters)
    dndx : np.ndarray
        XI-derivative of inverse metric factor pn
    dmde : np.ndarray
        ETA-derivative of inverse metric factor pm
        
    Examples:
    ---------
    >>> grid = get_roms_grid('my_grid.nc')
    >>> pm, pn, dndx, dmde = grid_metrics(grid)
    >>> print(f"Grid spacing ranges: {1/pm.max():.1f} - {1/pm.min():.1f} m")
    """
    if isinstance(G, str):
        G = get_roms_grid(G)
        
    spherical = G.get('spherical', True)
    
    # Get coordinates
    if great_circle and spherical:
        Xr = G.get('lon_rho', np.array([]))
        Yr = G.get('lat_rho', np.array([]))
        Xu = G.get('lon_u', np.array([]))
        Yu = G.get('lat_u', np.array([]))
        Xv = G.get('lon_v', np.array([]))
        Yv = G.get('lat_v', np.array([]))
    else:
        Xr = G.get('x_rho', np.array([]))
        Yr = G.get('y_rho', np.array([]))
        Xu = G.get('x_u', np.array([]))
        Yu = G.get('y_u', np.array([]))
        Xv = G.get('x_v', np.array([]))
        Yv = G.get('y_v', np.array([]))
    
    if len(Xr) == 0:
        raise ValueError("Grid coordinates not found")
    
    Lp, Mp = Xr.shape
    L = Lp - 1
    M = Mp - 1
    Lm = L - 1
    Mm = M - 1
    
    dx = np.zeros_like(Xr)
    dy = np.zeros_like(Xr)
    
    # Compute grid spacing
    if great_circle and spherical:
        # Great circle distances
        dx[1:L, :] = gcircle(Xu[0:Lm, :], Yu[0:Lm, :], Xu[1:L, :], Yu[1:L, :]) * 1000  # Convert km to m
        dx[0, :] = gcircle(Xr[0, :], Yr[0, :], Xu[0, :], Yu[0, :]) * 2000
        dx[Lp-1, :] = gcircle(Xu[L-1, :], Yu[L-1, :], Xr[Lp-1, :], Yr[Lp-1, :]) * 2000
        
        dy[:, 1:M] = gcircle(Xv[:, 0:Mm], Yv[:, 0:Mm], Xv[:, 1:M], Yv[:, 1:M]) * 1000
        dy[:, 0] = gcircle(Xr[:, 0], Yr[:, 0], Xv[:, 0], Yv[:, 0]) * 2000
        dy[:, Mp-1] = gcircle(Xv[:, M-1], Yv[:, M-1], Xr[:, Mp-1], Yr[:, Mp-1]) * 2000
    else:
        # Cartesian distances
        dx[1:L, :] = np.sqrt((Xu[1:L, :] - Xu[0:Lm, :])**2 + 
                            (Yu[1:L, :] - Yu[0:Lm, :])**2)
        dx[0, :] = np.sqrt((Xu[0, :] - Xr[0, :])**2 + 
                          (Yu[0, :] - Yr[0, :])**2) * 2.0
        dx[Lp-1, :] = np.sqrt((Xr[Lp-1, :] - Xu[L-1, :])**2 + 
                             (Yr[Lp-1, :] - Yu[L-1, :])**2) * 2.0
                             
        dy[:, 1:M] = np.sqrt((Xv[:, 1:M] - Xv[:, 0:Mm])**2 + 
                            (Yv[:, 1:M] - Yv[:, 0:Mm])**2)
        dy[:, 0] = np.sqrt((Xv[:, 0] - Xr[:, 0])**2 + 
                          (Yv[:, 0] - Yr[:, 0])**2) * 2.0
        dy[:, Mp-1] = np.sqrt((Xr[:, Mp-1] - Xv[:, M-1])**2 + 
                             (Yr[:, Mp-1] - Yv[:, M-1])**2) * 2.0
    
    # Compute inverse grid spacing metrics
    pm = 1.0 / dx
    pn = 1.0 / dy
    
    # Compute inverse metric derivatives
    dndx = np.zeros_like(Xr)
    dmde = np.zeros_like(Xr)
    
    # Check if grid is uniform
    uniform = (len(np.unique(pm.flatten())) == 1 and 
              len(np.unique(pn.flatten())) == 1)
    
    if not uniform:
        dndx[1:L, 1:M] = 0.5 * (1.0/pn[2:Lp, 1:M] - 1.0/pn[0:Lm, 1:M])
        dmde[1:L, 1:M] = 0.5 * (1.0/pm[1:L, 2:Mp] - 1.0/pm[1:L, 0:Mm])
    
    return pm, pn, dndx, dmde


def gcircle(lon1, lat1, lon2, lat2) -> np.ndarray:
    """
    Compute great circle distance between points using the Haversine formula.
    
    Parameters:
    -----------
    lon1, lat1 : float or np.ndarray
        First point coordinates (degrees)
    lon2, lat2 : float or np.ndarray 
        Second point coordinates (degrees)
        
    Returns:
    --------
    distance : float or np.ndarray
        Great circle distance in kilometers
        
    Examples:
    ---------
    >>> dist = gcircle(-122.0, 37.0, -121.0, 38.0)  # ~134 km
    >>> print(f"Distance: {dist:.1f} km")
    """
    # Convert inputs to numpy arrays
    lon1 = np.asarray(lon1)
    lat1 = np.asarray(lat1)
    lon2 = np.asarray(lon2)
    lat2 = np.asarray(lat2)
    # Convert to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    R = 6371.0
    
    return R * c


def inpolygon_numpy(xq: np.ndarray, yq: np.ndarray, 
                   xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """
    NumPy implementation of MATLAB's inpolygon function.
    
    Determines which query points are inside a polygon defined by vertices.
    Uses matplotlib's Path class for efficient point-in-polygon testing.
    
    Parameters:
    -----------
    xq, yq : np.ndarray
        Query points coordinates
    xv, yv : np.ndarray
        Polygon vertices coordinates (should form a closed polygon)
        
    Returns:
    --------
    in_poly : np.ndarray
        Boolean array indicating which points are inside polygon
        
    Examples:
    ---------
    >>> # Define a simple square polygon
    >>> xv = np.array([0, 1, 1, 0, 0])
    >>> yv = np.array([0, 0, 1, 1, 0])
    >>> # Test points
    >>> xq = np.array([0.5, 1.5, 0.5])
    >>> yq = np.array([0.5, 0.5, 1.5])
    >>> inside = inpolygon_numpy(xq, yq, xv, yv)
    >>> print(inside)  # [True, False, False]
    """
    try:
        from matplotlib.path import Path
    except ImportError:
        raise ImportError("matplotlib is required for inpolygon_numpy function")
    
    # Create polygon path
    polygon = Path(np.column_stack((xv, yv)))
    
    # Test which points are inside
    points = np.column_stack((xq.flatten(), yq.flatten()))
    in_poly = polygon.contains_points(points)
    
    return in_poly.reshape(xq.shape)


def extract_boundary_segments(perimeter_x: np.ndarray, perimeter_y: np.ndarray,
                             boundary_edges: list) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract boundary segments from perimeter coordinates.
    
    This function takes perimeter coordinates and extracts the individual
    boundary segments (western, southern, eastern, northern) based on
    the boundary edge definitions.
    
    Parameters:
    -----------
    perimeter_x, perimeter_y : np.ndarray
        Perimeter coordinates
    boundary_edges : list
        Boundary edge indices defining the segments
        
    Returns:
    --------
    boundaries : Dict[str, Dict[str, np.ndarray]]
        Boundary segment coordinates for each edge
        
    Notes:
    ------
    This is a placeholder implementation. The full version would need
    to analyze the perimeter structure and boundary edge definitions
    to properly extract each boundary segment.
    """
    
    boundaries = {
        'western': {'X': np.array([]), 'Y': np.array([])},
        'southern': {'X': np.array([]), 'Y': np.array([])},
        'eastern': {'X': np.array([]), 'Y': np.array([])},
        'northern': {'X': np.array([]), 'Y': np.array([])}
    }
    
    # This is a placeholder implementation
    # The full version would analyze perimeter_x, perimeter_y and boundary_edges
    # to extract the individual boundary segments
    
    return boundaries


def grids_structure(gnames: list) -> list:
    """
    Build ROMS nested grids structure array containing all variables
    associated with the application's horizontal and vertical grids.
    
    This function reads multiple ROMS grid files and creates a standardized
    structure array for use in nested grid applications.
    
    Parameters:
    -----------
    gnames : list
        List of ROMS Grid NetCDF file names containing all grid variables
    
    Returns:
    --------
    G : list
        Nested grids structure (list of grid dictionaries)
        
    Examples:
    ---------
    >>> grid_files = ['parent.nc', 'child1.nc', 'child2.nc']
    >>> G = grids_structure(grid_files)
    >>> print(f"Loaded {len(G)} grids")
    """
    # Initialize
    parent = ['parent_grid', 'parent_Imin', 'parent_Imax', 'parent_Jmin', 'parent_Jmax']
    ngrids = len(gnames)
    G = []
    
    # Get nested grid structures
    for n in range(ngrids):
        g = get_roms_grid(gnames[n])
        
        # Remove parent fields to have array of similar structures
        # (These will be handled separately in nested grid processing)
        for field in parent:
            g.pop(field, None)
        
        G.append(g)
    
    return G

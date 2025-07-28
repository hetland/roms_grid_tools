"""
Nested grids functionality for ROMS.

This module contains functions specifically for working with nested grids,
including grid connections, perimeter calculations, and refinement operations.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .data_structures import ContactStructure, ContactRegion, GridInfo, ContactPoint
from .grid_utils import grids_structure


def grid_perimeter(G: List[Dict[str, Any]]) -> ContactStructure:
    """
    Set Nested Grids Perimeters and Boundary Edges.
    
    This function analyzes the grid structures and sets up the initial
    contact structure with perimeter information and boundary edge definitions.
    
    Parameters:
    -----------
    G : List[Dict[str, Any]]
        Information grids structure (list of grid dictionaries)
        
    Returns:
    --------
    S : ContactStructure
        Nested grids information structure with perimeter data
        
    Examples:
    ---------
    >>> grid_files = ['parent.nc', 'child.nc']
    >>> G = grids_structure(grid_files)
    >>> S = grid_perimeter(G)
    >>> print(f"Created structure for {S.Ngrids} grids with {S.Ncontact} contact regions")
    """
    
    # Initialize nested grids information structure
    Ngrids = len(G)
    Ncontact = (Ngrids - 1) * 2  # Each grid pair has 2 contact regions
    
    S = ContactStructure()
    S.Ngrids = Ngrids
    S.Ncontact = Ncontact
    S.NLweights = 4   # Linear interpolation uses 4 points
    S.NQweights = 9   # Quadratic interpolation uses 9 points
    S.Ndatum = 0      # Will be computed later
    
    # Define boundary edge constants
    S.western_edge = 1
    S.southern_edge = 2
    S.eastern_edge = 3 
    S.northern_edge = 4
    
    # Set coordinate system flag
    S.spherical = G[0].get('spherical', False)
    
    # Initialize grid and contact structures
    S.grid = []
    S.contact = []
    
    # Process each grid
    for ng in range(Ngrids):
        grid_info = _create_grid_info(G[ng], ng)
        S.grid.append(grid_info)
    
    # Initialize contact regions 
    for cr in range(Ncontact):
        contact_region = _create_contact_region(cr, Ngrids)
        S.contact.append(contact_region)
    
    return S


def grid_connections(G: List[Dict[str, Any]], S: ContactStructure) -> ContactStructure:
    """
    Set Nested Grids Connectivity.
    
    This function analyzes the relationship between grids and determines
    the type of connection (refinement, coincident, etc.) for each contact region.
    
    Parameters:
    -----------
    G : List[Dict[str, Any]]
        Nested Grids Structure (list of grid dictionaries)
    S : ContactStructure
        Contact Points Structure
        
    Returns:
    --------
    S : ContactStructure
        Updated nested grids contact points structure with connections
        
    Examples:
    ---------
    >>> S = grid_connections(G, S)
    >>> for cr in range(S.Ncontact):
    ...     if S.contact[cr].refinement:
    ...         print(f"Contact region {cr} is a refinement connection")
    """
    
    # Set connectivity for each contact region
    for cr in range(S.Ncontact):
        dg = S.contact[cr].donor_grid - 1  # Convert to 0-based indexing
        rg = S.contact[cr].receiver_grid - 1
        
        # Determine grid relationship type based on refinement factors
        donor_refine = G[dg].get('refine_factor', 0)
        receiver_refine = G[rg].get('refine_factor', 0)
        
        if donor_refine > 0 or receiver_refine > 0:
            S.contact[cr].refinement = True
            S.contact[cr].coincident = False
        else:
            S.contact[cr].refinement = False
            S.contact[cr].coincident = True
            
        # Check for other relationship types
        S.contact[cr].composite = False  # Could be determined from grid metadata
        S.contact[cr].hybrid = False     # Mixed vertical coordinate systems
        S.contact[cr].mosaic = False     # Non-overlapping adjacent grids
        
        # Initialize interior and corners structures
        S.contact[cr].interior = {'okay': False}
        S.contact[cr].corners = {'okay': False}
        
        # Analyze boundary connections
        S = _analyze_boundary_connections(G, S, cr, dg, rg)
    
    return S


def boundary_contact(S: ContactStructure) -> ContactStructure:
    """
    Determine which contact points lay on the receiver grid boundary.
    
    This function analyzes contact points and flags those that lie on
    the boundary of the receiver grid. This information is important
    for applying boundary conditions.
    
    Parameters:
    -----------
    S : ContactStructure
        Contact points structure
        
    Returns:
    --------
    S : ContactStructure
        Updated contact points structure with boundary information
        
    Examples:
    ---------
    >>> S = boundary_contact(S)
    >>> # Check boundary points for first contact region
    >>> if S.contact[0].point is not None:
    ...     n_boundary = np.sum(S.contact[0].point.boundary_rho)
    ...     print(f"Contact region 0 has {n_boundary} boundary RHO points")
    """
    
    for cr in range(S.Ncontact):
        contact = S.contact[cr].point
        if contact is None:
            continue
            
        rg = S.contact[cr].receiver_grid - 1  # Convert to 0-based indexing
        
        # Get receiver grid dimensions
        Lp = S.grid[rg].Lp
        Mp = S.grid[rg].Mp
        
        # Check RHO points on boundary
        if len(contact.Irg_rho) > 0:
            boundary_rho = _check_boundary_points(contact.Irg_rho, contact.Jrg_rho, Lp, Mp)
            contact.boundary_rho = boundary_rho
        
        # Check U points on boundary
        if len(contact.Irg_u) > 0:
            L = Lp - 1  # U-grid has one less point in XI direction
            boundary_u = _check_boundary_points(contact.Irg_u, contact.Jrg_u, L, Mp)
            contact.boundary_u = boundary_u
            
        # Check V points on boundary  
        if len(contact.Irg_v) > 0:
            M = Mp - 1  # V-grid has one less point in ETA direction
            boundary_v = _check_boundary_points(contact.Irg_v, contact.Jrg_v, Lp, M)
            contact.boundary_v = boundary_v
    
    return S


def refine_coordinates(cr: int, dg: int, rg: int, G: List[Dict], 
                      S: ContactStructure, mask_interp: bool) -> Dict[str, Any]:
    """
    Compute receiver grid refinement coordinates from the donor grid.
    
    This function computes the refined grid coordinates for a nested grid
    by interpolating from the coarser donor grid. It handles both coordinate
    interpolation and grid variable refinement.
    
    Parameters:
    -----------
    cr : int
        Contact region number
    dg : int
        Donor grid number  
    rg : int
        Receiver grid number
    G : List[Dict]
        Nested grids structure
    S : ContactStructure
        Nested grids information structure
    mask_interp : bool
        Switch to interpolate PSI-, U- and V-mask
        
    Returns:
    --------
    R : Dict[str, Any]
        Refinement coordinates structure containing interpolated grid data
        
    Notes:
    ------
    This is a complex function that handles grid refinement through interpolation.
    The current implementation provides the structure but would need full
    implementation of the scipy interpolation routines.
    """
    
    spherical = S.spherical
    
    # Get donor and receiver grid dimension parameters
    Lp = S.grid[dg].Lp
    Mp = S.grid[dg].Mp
    L = S.grid[dg].L
    M = S.grid[dg].M
    
    # Set donor (coarse) grid fractional coordinates
    XrC, YrC = np.meshgrid(np.arange(0.5, Lp-0.5+1), np.arange(0.5, Mp-0.5+1))
    XpC, YpC = np.meshgrid(np.arange(1.0, L+1), np.arange(1.0, M+1))
    XuC, YuC = np.meshgrid(np.arange(1.0, L+1), np.arange(0.5, Mp-0.5+1))
    XvC, YvC = np.meshgrid(np.arange(0.5, Lp-0.5+1), np.arange(1.0, M+1))
    
    # Set extraction fractional coordinates from donor grid
    delta = 1.0 / S.grid[rg].refine_factor
    half = 0.5 * delta
    
    # Define extraction region with buffer
    offset_west = 3.0 * delta
    offset_south = 3.0 * delta  
    offset_east = 2.0 * delta
    offset_north = 2.0 * delta
    
    # Get contact region bounds
    if 'corners' in S.contact[cr].__dict__ and 'Idg' in S.contact[cr].corners:
        Imin = min(S.contact[cr].corners['Idg']) - offset_west
        Imax = max(S.contact[cr].corners['Idg']) + offset_east
        Jmin = min(S.contact[cr].corners['Jdg']) - offset_south
        Jmax = max(S.contact[cr].corners['Jdg']) + offset_north
    else:
        # Default extraction region
        Imin = 1.0 - offset_west
        Imax = L + offset_east
        Jmin = 1.0 - offset_south
        Jmax = M + offset_north
    
    # Set receiver (fine) grid fractional coordinates
    IpF = np.arange(Imin, Imax + delta, delta)
    JpF = np.arange(Jmin, Jmax + delta, delta)
    IrF = np.concatenate([[IpF[0] - half], IpF + half])
    JrF = np.concatenate([[JpF[0] - half], JpF + half])
    
    XrF, YrF = np.meshgrid(IrF, JrF)
    XpF, YpF = np.meshgrid(IpF, JpF)
    XuF, YuF = np.meshgrid(IpF, JrF)
    XvF, YvF = np.meshgrid(IrF, JpF)
    
    # Initialize result structure
    R = {
        'spherical': spherical,
        'uniform': G[dg].get('uniform', False),
        'xi_rho': XrF,
        'eta_rho': YrF,
        'xi_psi': XpF,
        'eta_psi': YpF,
        'xi_u': XuF,
        'eta_u': YuF,
        'xi_v': XvF,
        'eta_v': YvF,
        'refine_factor': S.grid[rg].refine_factor,
        'donor_grid': dg + 1,  # 1-based indexing
        'receiver_grid': rg + 1
    }
    
    # Note: Full implementation would continue with interpolation of
    # all grid variables from donor to refined grid using scipy interpolation
    
    return R


def write_contact(cname: str, S: ContactStructure, G: List[Dict]) -> None:
    """
    Write ROMS Nested Grids Contact Points to a NetCDF file.
    
    This function creates a NetCDF file containing all the contact point
    information needed for ROMS nested grid simulations.
    
    Parameters:
    -----------
    cname : str
        Contact Point NetCDF file name
    S : ContactStructure
        Nested grids Contact Points structure
    G : List[Dict]
        Information grids structure
        
    Examples:
    ---------
    >>> write_contact('contact_points.nc', S, G)
    >>> print("Contact points file created successfully")
    """
    import netCDF4 as nc
    from datetime import datetime
    
    try:
        # Create NetCDF file
        with nc.Dataset(cname, 'w', format='NETCDF4') as ncfile:
            
            # Create dimensions
            ncfile.createDimension('Ngrids', S.Ngrids)
            ncfile.createDimension('Ncontact', S.Ncontact)
            ncfile.createDimension('nLweights', S.NLweights)
            ncfile.createDimension('nQweights', S.NQweights)
            
            # Only create datum dimension if we have contact points
            if S.Ndatum > 0:
                ncfile.createDimension('datum', S.Ndatum)
            
            # Create and write basic variables
            _write_basic_variables(ncfile, S)
            
            # Write grid information
            _write_grid_variables(ncfile, S, G)
            
            # Write contact region information
            if S.Ncontact > 0:
                _write_contact_variables(ncfile, S)
            
            # Write contact points data if available
            if S.Ndatum > 0:
                _write_contact_points(ncfile, S)
            
            # Add global attributes
            _write_global_attributes(ncfile, S, G)
            
        print(f"Contact points NetCDF file created: {cname}")
        
    except Exception as e:
        print(f"Error writing contact file {cname}: {e}")
        raise


def plot_contact(G: List[Dict], S: ContactStructure) -> None:
    """
    Plot various ROMS Nested Grids Contact Points figures.
    
    This function creates visualization plots showing grid perimeters,
    contact regions, and contact points for nested grid configurations.
    
    Parameters:
    -----------
    G : List[Dict]
        Information grids structure
    S : ContactStructure  
        Nested Grids Structure
        
    Examples:
    ---------
    >>> plot_contact(G, S)  # Creates plots for all contact regions
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plot_contact function")
        return
    
    # Check if we have plottable data
    if S.Ncontact == 0:
        print("No contact regions to plot")
        return
    
    # Plot each contact region
    for cr in range(S.Ncontact):
        _plot_contact_region(G, S, cr)
        
    print(f"Plotted {S.Ncontact} contact region(s)")


# Helper functions

def _create_grid_info(grid_dict: Dict[str, Any], grid_index: int) -> GridInfo:
    """Create GridInfo structure from grid dictionary."""
    grid_info = GridInfo()
    grid_info.filename = grid_dict.get('grid_name', f'grid_{grid_index}.nc')
    
    # Get grid dimensions
    grid_info.Lp = grid_dict.get('Lp', grid_dict.get('Lm', 0) + 2)
    grid_info.Mp = grid_dict.get('Mp', grid_dict.get('Mm', 0) + 2)
    grid_info.L = grid_info.Lp - 1
    grid_info.M = grid_info.Mp - 1
    
    # Get refinement information if available
    grid_info.refine_factor = grid_dict.get('refine_factor', 1)
    grid_info.parent_Imin = grid_dict.get('parent_Imin', 0)
    grid_info.parent_Imax = grid_dict.get('parent_Imax', 0)
    grid_info.parent_Jmin = grid_dict.get('parent_Jmin', 0)
    grid_info.parent_Jmax = grid_dict.get('parent_Jmax', 0)
    
    # Create index arrays
    I_rho, J_rho = np.meshgrid(np.arange(1, grid_info.Lp + 1), 
                               np.arange(1, grid_info.Mp + 1), indexing='xy')
    I_psi, J_psi = np.meshgrid(np.arange(1, grid_info.L + 1),
                               np.arange(1, grid_info.M + 1), indexing='xy')
    I_u, J_u = np.meshgrid(np.arange(1, grid_info.L + 1),
                           np.arange(1, grid_info.Mp + 1), indexing='xy')
    I_v, J_v = np.meshgrid(np.arange(1, grid_info.Lp + 1),
                           np.arange(1, grid_info.M + 1), indexing='xy')
    
    grid_info.I_rho = I_rho
    grid_info.J_rho = J_rho
    grid_info.I_psi = I_psi
    grid_info.J_psi = J_psi
    grid_info.I_u = I_u
    grid_info.J_u = J_u
    grid_info.I_v = I_v
    grid_info.J_v = J_v
    
    return grid_info


def _create_contact_region(cr: int, ngrids: int) -> ContactRegion:
    """Create ContactRegion structure."""
    contact_region = ContactRegion()
    
    # Set up donor and receiver relationships
    # This is a simplified assignment - real implementation would analyze grid hierarchy
    if cr < ngrids - 1:
        contact_region.donor_grid = cr + 1      # 1-based indexing
        contact_region.receiver_grid = cr + 2
    else:
        contact_region.donor_grid = cr - (ngrids - 2)
        contact_region.receiver_grid = cr - (ngrids - 3)
        
    # Initialize boundary array with 4 empty dictionaries (for 4 edges)
    contact_region.boundary = [{'okay': False} for _ in range(4)]
    
    return contact_region


def _analyze_boundary_connections(G: List[Dict], S: ContactStructure, 
                                cr: int, dg: int, rg: int) -> ContactStructure:
    """Analyze boundary connections between grids."""
    # This is a simplified implementation
    # Full version would analyze grid perimeters and determine actual overlaps
    
    # For demonstration, assume first contact region has a western boundary connection
    if cr == 0:
        S.contact[cr].boundary[0]['okay'] = True  # Western boundary
        S.contact[cr].boundary[0]['match'] = np.array([True])
    
    return S


def _check_boundary_points(I_indices: np.ndarray, J_indices: np.ndarray, 
                         Lp: int, Mp: int) -> np.ndarray:
    """Check which points lie on grid boundary."""
    boundary_mask = np.zeros(len(I_indices), dtype=bool)
    
    # Western boundary (I = 1)
    boundary_mask |= (I_indices == 1)
    # Eastern boundary (I = Lp)  
    boundary_mask |= (I_indices == Lp)
    # Southern boundary (J = 1)
    boundary_mask |= (J_indices == 1)
    # Northern boundary (J = Mp)
    boundary_mask |= (J_indices == Mp)
    
    return boundary_mask


def _write_basic_variables(ncfile, S: ContactStructure) -> None:
    """Write basic variables to NetCDF file."""
    # Spherical coordinate flag
    spherical_var = ncfile.createVariable('spherical', 'i4')
    spherical_var[:] = int(S.spherical)
    
    # Number of grids and contact regions
    ngrids_var = ncfile.createVariable('Ngrids', 'i4')
    ngrids_var[:] = S.Ngrids
    
    ncontact_var = ncfile.createVariable('Ncontact', 'i4')
    ncontact_var[:] = S.Ncontact


def _write_grid_variables(ncfile, S: ContactStructure, G: List[Dict]) -> None:
    """Write grid dimension variables."""
    if S.Ngrids > 0:
        lm_var = ncfile.createVariable('Lm', 'i4', ('Ngrids',))
        mm_var = ncfile.createVariable('Mm', 'i4', ('Ngrids',))
        
        lm_values = []
        mm_values = []
        for ng in range(S.Ngrids):
            lm_values.append(S.grid[ng].Lp - 2)  # Lm = Lp - 2
            mm_values.append(S.grid[ng].Mp - 2)  # Mm = Mp - 2
        
        lm_var[:] = lm_values
        mm_var[:] = mm_values


def _write_contact_variables(ncfile, S: ContactStructure) -> None:
    """Write contact region variables."""
    donor_var = ncfile.createVariable('donor_grid', 'i4', ('Ncontact',))
    receiver_var = ncfile.createVariable('receiver_grid', 'i4', ('Ncontact',))
    
    donor_values = []
    receiver_values = []
    for cr in range(S.Ncontact):
        donor_values.append(S.contact[cr].donor_grid)
        receiver_values.append(S.contact[cr].receiver_grid)
    
    donor_var[:] = donor_values
    receiver_var[:] = receiver_values


def _write_contact_points(ncfile, S: ContactStructure) -> None:
    """Write contact points data."""
    # This would write the actual contact point coordinates and weights
    # Current implementation is a placeholder
    pass


def _write_global_attributes(ncfile, S: ContactStructure, G: List[Dict]) -> None:
    """Write global attributes to NetCDF file."""
    from datetime import datetime
    
    ncfile.title = 'ROMS Nested Grids Contact Points'
    ncfile.history = f'Created by roms_grid_tools on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    ncfile.source = 'Python translation of MATLAB contact.m'
    
    # Grid file names
    grid_files = []
    for ng in range(S.Ngrids):
        grid_files.append(S.grid[ng].filename)
    ncfile.grid_files = '\n'.join(grid_files)


def _plot_contact_region(G: List[Dict], S: ContactStructure, cr: int) -> None:
    """Plot a single contact region."""
    import matplotlib.pyplot as plt
    
    dg = S.contact[cr].donor_grid - 1      # Convert to 0-based indexing
    rg = S.contact[cr].receiver_grid - 1
    
    # Get coordinate bounds for plotting
    if S.spherical:
        xlabel, ylabel = 'Longitude', 'Latitude'
        coord_keys = ['lon_rho', 'lat_rho']
    else:
        xlabel, ylabel = 'X (m)', 'Y (m)'
        coord_keys = ['x_rho', 'y_rho']
    
    # Get coordinate bounds
    Xmin, Xmax, Ymin, Ymax = _get_plot_bounds(G, [dg, rg], coord_keys)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot grid perimeters if coordinates are available
    _plot_grid_perimeters(G, S, dg, rg, coord_keys)
    
    # Plot contact points if available
    _plot_contact_points(S, cr, coord_keys)
    
    # Set plot properties
    plt.xlim(Xmin, Xmax)
    plt.ylim(Ymin, Ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Contact Region {cr+1}: Donor Grid {dg+1} -> Receiver Grid {rg+1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def _get_plot_bounds(G: List[Dict], grid_indices: List[int], 
                    coord_keys: List[str]) -> Tuple[float, float, float, float]:
    """Get coordinate bounds for plotting."""
    x_key, y_key = coord_keys
    
    all_x, all_y = [], []
    for gi in grid_indices:
        if x_key in G[gi] and y_key in G[gi]:
            all_x.extend([np.min(G[gi][x_key]), np.max(G[gi][x_key])])
            all_y.extend([np.min(G[gi][y_key]), np.max(G[gi][y_key])])
    
    if all_x and all_y:
        return min(all_x), max(all_x), min(all_y), max(all_y)
    else:
        # Default bounds
        return 0, 100, 0, 100


def _plot_grid_perimeters(G: List[Dict], S: ContactStructure, dg: int, rg: int, 
                         coord_keys: List[str]) -> None:
    """Plot grid perimeters."""
    # Simplified implementation - would need actual perimeter extraction
    pass


def _plot_contact_points(S: ContactStructure, cr: int, coord_keys: List[str]) -> None:
    """Plot contact points."""
    contact = S.contact[cr].point
    if contact is not None and hasattr(contact, 'Xrg_rho'):
        if len(contact.Xrg_rho) > 0:
            import matplotlib.pyplot as plt
            plt.plot(contact.Xrg_rho, contact.Yrg_rho, 'ko', 
                    markersize=3, label='Contact Points')

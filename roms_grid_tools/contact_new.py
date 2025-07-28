"""
contact.py - Main contact points function for ROMS nested grids

Sets Contact Points between ROMS nested Grids.

This function sets contact points in the overlapping contact
regions between nested grids. The order of nested grid file
names in input list (gnames) is important. Set the
file names in the order of nesting layers and time-stepping
in ROMS.

Author: Translated from MATLAB by Python conversion
Original MATLAB version: Copyright (c) 2002-2023 The ROMS/TOMS Group
"""

import numpy as np
from typing import List, Dict, Tuple, Any

# Import from local modules
from .data_structures import ContactStructure, ContactPoint
from .grid_utils import grids_structure
from .nested_grids import (
    grid_perimeter, grid_connections, boundary_contact, 
    write_contact, plot_contact
)
from .weights import linear_weights, quadratic_weights


def contact(gnames: List[str], cname: str, 
           lmask: bool = False, 
           mask_interp: bool = False, 
           lplot: bool = False) -> Tuple[ContactStructure, List[Dict[str, Any]]]:
    """
    Sets Contact Points between ROMS nested Grids.
    
    Parameters:
    -----------
    gnames : List[str]
        Input Grid NetCDF file names
    cname : str
        Output Contact Points NetCDF file name
    lmask : bool, optional
        Switch to remove Contact Points over land (default False)
    mask_interp : bool, optional
        Switch to interpolate PSI-, U- and V-masks (True) or
        computed from interpolated RHO-mask (False) (default False)
    lplot : bool, optional
        Switch to plot various Contact Points figures (default False)
        
    Returns:
    --------
    S : ContactStructure
        Nested grids Contact Points structure
    G : List[Dict[str, Any]]
        Nested grids structure
        
    Examples:
    ---------
    >>> grid_files = ['parent.nc', 'child1.nc', 'child2.nc']
    >>> S, G = contact(grid_files, 'contact.nc', lmask=True, lplot=True)
    >>> print(f"Created contact points for {S.Ngrids} grids")
    """
    
    ngrids = len(gnames)
    ncontact = (ngrids - 1) * 2
    
    # Get nested grids information and set perimeters and boundary edges
    G = grids_structure(gnames)
    
    # Set nested grids perimeters and boundary edges
    S = grid_perimeter(G)
    
    # Set up nested grid connections
    S = grid_connections(G, S)
    
    # Set contact points in each contact region
    ndatum = 0
    
    for cr in range(ncontact):
        dg = S.contact[cr].donor_grid
        rg = S.contact[cr].receiver_grid
        
        if S.contact[cr].coincident:
            P = coincident(cr, dg, rg, lmask, G, S)
            S.contact[cr].point = P
        elif S.contact[cr].refinement:
            P, R = refinement(cr, dg, rg, lmask, G, S, mask_interp)
            S.contact[cr].point = P
            if len(S.refined) <= cr:
                S.refined.extend([{}] * (cr + 1 - len(S.refined)))
            S.refined[cr] = R
            
        # Only count points if they exist
        if S.contact[cr].point is not None:
            point = S.contact[cr].point
            rho_count = len(getattr(point, 'Irg_rho', [])) 
            u_count = len(getattr(point, 'Irg_u', []))
            v_count = len(getattr(point, 'Irg_v', []))
            ndatum += (rho_count + u_count + v_count)
    
    S.Ndatum = ndatum
    
    # Determine which contact points lay on the receiver grid boundary
    S = boundary_contact(S)
    
    # Set contact points horizontal interpolation weights
    impose_mask = False
    S = hweights(G, S, impose_mask)
    
    # Create and write out Contact Point data into output NetCDF file
    write_contact(cname, S, G)
    
    # Plot contact areas and contact points
    if lplot:
        plot_contact(G, S)
    
    return S, G


def coincident(cr: int, dg: int, rg: int, lmask: bool, 
              G: List[Dict[str, Any]], S: ContactStructure) -> ContactPoint:
    """
    Sets contact points for coincident grids.
    
    Parameters:
    -----------
    cr : int
        Contact region number
    dg : int
        Donor grid number
    rg : int
        Receiver grid number  
    lmask : bool
        Switch to remove contact points on land
    G : List[Dict[str, Any]]
        Nested grids structure
    S : ContactStructure
        Nested grids information structure
        
    Returns:
    --------
    C : ContactPoint
        Contact points structure
    """
    
    # Initialize
    iwest = S.western_edge
    isouth = S.southern_edge
    ieast = S.eastern_edge
    inorth = S.northern_edge
    
    Lpdg = S.grid[dg].Lp
    Mpdg = S.grid[dg].Mp
    Lprg = S.grid[rg].Lp
    Mprg = S.grid[rg].Mp
    
    Ldg = S.grid[dg].L
    Mdg = S.grid[dg].M
    Lrg = S.grid[rg].L
    Mrg = S.grid[rg].M
    
    spherical = S.spherical
    
    C = ContactPoint()
    W = [ContactPoint() for _ in range(4)]  # Working arrays for each boundary
    
    # Process each boundary
    # Western boundary
    if S.contact[cr].boundary[iwest-1].get('okay', False):
        _process_western_boundary(W[iwest-1], dg, rg, G, S, spherical)
    
    # Southern boundary  
    if S.contact[cr].boundary[isouth-1].get('okay', False):
        _process_southern_boundary(W[isouth-1], dg, rg, G, S, spherical)
        
    # Eastern boundary
    if S.contact[cr].boundary[ieast-1].get('okay', False):
        _process_eastern_boundary(W[ieast-1], dg, rg, G, S, spherical)
        
    # Northern boundary
    if S.contact[cr].boundary[inorth-1].get('okay', False):
        _process_northern_boundary(W[inorth-1], dg, rg, G, S, spherical)
    
    # Convert contact data to vectors
    for ib in range(4):
        if S.contact[cr].boundary[ib].get('okay', False):
            C = _merge_boundary_data(C, W[ib], lmask)
            break  # Currently assuming single boundary connection
    
    return C


def refinement(cr: int, dg: int, rg: int, lmask: bool,
              G: List[Dict[str, Any]], S: ContactStructure, 
              mask_interp: bool) -> Tuple[ContactPoint, Dict[str, Any]]:
    """
    Sets contact points for refinement grids.
    
    Parameters:
    -----------
    cr : int
        Contact region number
    dg : int
        Donor grid number
    rg : int  
        Receiver grid number
    lmask : bool
        Switch to remove contact points on land
    G : List[Dict[str, Any]]
        Nested grids structure
    S : ContactStructure
        Nested grids information structure
    mask_interp : bool
        Switch to interpolate masks
        
    Returns:
    --------
    C : ContactPoint
        Contact points structure
    R : Dict[str, Any]
        Refinement structure
    """
    
    # Get refined coordinates
    from .nested_grids import refine_coordinates
    R = refine_coordinates(cr, dg, rg, G, S, mask_interp)
    
    # Extract contact region from refined grid
    C = _extract_contact_region(R, cr, dg, rg, lmask, G, S)
    
    return C, R


def hweights(G: List[Dict[str, Any]], S: ContactStructure, impose_mask: bool) -> ContactStructure:
    """
    Compute horizontal interpolation weights for contact points.
    
    Parameters:
    -----------
    G : List[Dict[str, Any]]
        Grid structures
    S : ContactStructure
        Contact structure
    impose_mask : bool
        Whether to impose land/sea masking
        
    Returns:
    --------
    S : ContactStructure
        Updated contact structure with weights
    """
    
    # Initialize weight structures
    S.Lweights = []
    S.Qweights = []
    
    for cr in range(S.Ncontact):
        dg = S.contact[cr].donor_grid - 1  # Convert to 0-based
        
        # Get contact points
        contact = S.contact[cr].point
        if contact is None:
            S.Lweights.append({})
            S.Qweights.append({})
            continue
            
        # Get donor grid mask
        mask = G[dg].get('mask_rho', None) if impose_mask else None
        
        # Compute linear weights for RHO points
        if len(contact.Irg_rho) > 0:
            W_rho = linear_weights(contact.xrg_rho, contact.erg_rho,
                                 contact.Idg_rho, contact.Jdg_rho,
                                 impose_mask, mask)
            QW_rho = quadratic_weights(contact.xrg_rho, contact.erg_rho,
                                     contact.Idg_rho, contact.Jdg_rho,
                                     impose_mask, mask)
        else:
            W_rho = np.array([]).reshape(4, 0)
            QW_rho = np.array([]).reshape(9, 0)
        
        # Compute linear weights for U points
        if len(contact.Irg_u) > 0:
            W_u = linear_weights(contact.xrg_u, contact.erg_u,
                               contact.Idg_u, contact.Jdg_u,
                               impose_mask, mask)
            QW_u = quadratic_weights(contact.xrg_u, contact.erg_u,
                                   contact.Idg_u, contact.Jdg_u,
                                   impose_mask, mask)
        else:
            W_u = np.array([]).reshape(4, 0)
            QW_u = np.array([]).reshape(9, 0)
            
        # Compute linear weights for V points
        if len(contact.Irg_v) > 0:
            W_v = linear_weights(contact.xrg_v, contact.erg_v,
                               contact.Idg_v, contact.Jdg_v,
                               impose_mask, mask)
            QW_v = quadratic_weights(contact.xrg_v, contact.erg_v,
                                   contact.Idg_v, contact.Jdg_v,
                                   impose_mask, mask)
        else:
            W_v = np.array([]).reshape(4, 0)
            QW_v = np.array([]).reshape(9, 0)
        
        # Store weights
        S.Lweights.append({
            'H_rho': W_rho,
            'H_u': W_u,
            'H_v': W_v
        })
        
        S.Qweights.append({
            'H_rho': QW_rho,
            'H_u': QW_u,
            'H_v': QW_v
        })
    
    return S


# Helper functions (simplified implementations)

def _process_western_boundary(W: ContactPoint, dg: int, rg: int, 
                             G: List[Dict[str, Any]], S: ContactStructure, spherical: bool):
    """Process western boundary contact points"""
    # Simplified implementation
    # Full version would extract actual boundary coordinates
    W.Xrg_rho = np.array([1.0, 1.0, 1.0])  # Example western boundary
    W.Yrg_rho = np.array([1.0, 2.0, 3.0])
    W.Idg_rho = np.array([1, 1, 1])
    W.Jdg_rho = np.array([1, 2, 3])


def _process_southern_boundary(W: ContactPoint, dg: int, rg: int, 
                              G: List[Dict[str, Any]], S: ContactStructure, spherical: bool):
    """Process southern boundary contact points"""
    # Simplified implementation
    W.Xrg_rho = np.array([1.0, 2.0, 3.0])  # Example southern boundary
    W.Yrg_rho = np.array([1.0, 1.0, 1.0])
    W.Idg_rho = np.array([1, 2, 3])
    W.Jdg_rho = np.array([1, 1, 1])


def _process_eastern_boundary(W: ContactPoint, dg: int, rg: int, 
                             G: List[Dict[str, Any]], S: ContactStructure, spherical: bool):
    """Process eastern boundary contact points"""
    # Simplified implementation
    Lp = S.grid[rg].Lp
    W.Xrg_rho = np.array([Lp, Lp, Lp])  # Example eastern boundary
    W.Yrg_rho = np.array([1.0, 2.0, 3.0])
    W.Idg_rho = np.array([Lp, Lp, Lp])
    W.Jdg_rho = np.array([1, 2, 3])


def _process_northern_boundary(W: ContactPoint, dg: int, rg: int, 
                              G: List[Dict[str, Any]], S: ContactStructure, spherical: bool):
    """Process northern boundary contact points"""
    # Simplified implementation
    Mp = S.grid[rg].Mp
    W.Xrg_rho = np.array([1.0, 2.0, 3.0])  # Example northern boundary
    W.Yrg_rho = np.array([Mp, Mp, Mp])
    W.Idg_rho = np.array([1, 2, 3])
    W.Jdg_rho = np.array([Mp, Mp, Mp])


def _merge_boundary_data(C: ContactPoint, W: ContactPoint, lmask: bool) -> ContactPoint:
    """Merge boundary data into contact structure"""
    # Copy contact point data from working structure to contact structure
    if hasattr(W, 'Xrg_rho') and len(W.Xrg_rho) > 0:
        C.Xrg_rho = np.concatenate([C.Xrg_rho, W.Xrg_rho]) if len(C.Xrg_rho) > 0 else W.Xrg_rho
        C.Yrg_rho = np.concatenate([C.Yrg_rho, W.Yrg_rho]) if len(C.Yrg_rho) > 0 else W.Yrg_rho
        C.Idg_rho = np.concatenate([C.Idg_rho, W.Idg_rho]) if len(C.Idg_rho) > 0 else W.Idg_rho
        C.Jdg_rho = np.concatenate([C.Jdg_rho, W.Jdg_rho]) if len(C.Jdg_rho) > 0 else W.Jdg_rho
        
        # Set other arrays to matching sizes for consistency
        npoints = len(C.Xrg_rho)
        C.xrg_rho = np.arange(npoints, dtype=float)
        C.erg_rho = np.arange(npoints, dtype=float)
        C.Irg_rho = np.arange(1, npoints+1, dtype=int)
        C.Jrg_rho = np.arange(1, npoints+1, dtype=int)
        
        # Apply land mask if requested
        if lmask and hasattr(W, 'mask_rho'):
            valid_mask = W.mask_rho > 0
            C.Xrg_rho = C.Xrg_rho[valid_mask]
            C.Yrg_rho = C.Yrg_rho[valid_mask]
            C.Idg_rho = C.Idg_rho[valid_mask]
            C.Jdg_rho = C.Jdg_rho[valid_mask]
    
    return C


def _extract_contact_region(R: Dict[str, Any], cr: int, dg: int, rg: int, lmask: bool, 
                           G: List[Dict[str, Any]], S: ContactStructure) -> ContactPoint:
    """Extract contact region from refined grid"""
    # Simplified implementation
    C = ContactPoint()
    
    # Get refinement grid coordinates
    if S.spherical:
        Xrg = G[rg].get('lon_rho', np.array([]))
        Yrg = G[rg].get('lat_rho', np.array([]))
        Xdg = G[dg].get('lon_rho', np.array([]))
        Ydg = G[dg].get('lat_rho', np.array([]))
    else:
        Xrg = G[rg].get('x_rho', np.array([]))
        Yrg = G[rg].get('y_rho', np.array([]))
        Xdg = G[dg].get('x_rho', np.array([]))
        Ydg = G[dg].get('y_rho', np.array([]))
    
    # Find overlap region (simplified implementation)
    if len(Xrg) > 0 and len(Xdg) > 0:
        # This is a simplified extraction - full implementation would
        # determine exact overlap regions and contact points
        overlap_points = min(len(Xrg.flatten()), 100)  # Limit for demo
        
        C.Xrg_rho = Xrg.flatten()[:overlap_points]
        C.Yrg_rho = Yrg.flatten()[:overlap_points]
        C.Idg_rho = np.ones(overlap_points, dtype=int)
        C.Jdg_rho = np.ones(overlap_points, dtype=int)
        
        # Set other arrays to matching sizes
        C.xrg_rho = np.arange(overlap_points, dtype=float)
        C.erg_rho = np.arange(overlap_points, dtype=float)
        C.Irg_rho = np.arange(overlap_points, dtype=int)
        C.Jrg_rho = np.arange(overlap_points, dtype=int)
    
    return C


if __name__ == "__main__":
    # Example usage
    print("Contact point generator for ROMS nested grids")
    print("This is a Python translation of the MATLAB contact.m function")
    print("Use the main contact() function to generate contact points")

"""
Interpolation weights for ROMS nested grids.

This module contains functions for computing horizontal interpolation weights
used in nested grid contact point calculations. Supports both linear and
quadratic interpolation schemes.
"""

import numpy as np
from typing import Optional


def linear_weights(xrg: np.ndarray, erg: np.ndarray, 
                  idg: np.ndarray, jdg: np.ndarray,
                  impose_mask: bool = False,
                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute linear horizontal interpolation weights for contact points.
    
    This function computes bilinear interpolation weights for interpolating
    from donor grid points to receiver grid contact points. The weights
    are computed for a 4-point stencil around each contact point.
    
    Parameters:
    -----------
    xrg : np.ndarray
        Receiver grid XI-coordinates (fractional grid indices)
    erg : np.ndarray
        Receiver grid ETA-coordinates (fractional grid indices)
    idg : np.ndarray
        Donor grid I-indices (integer grid indices)
    jdg : np.ndarray
        Donor grid J-indices (integer grid indices)
    impose_mask : bool, optional
        Whether to impose land/sea masking (default False)
    mask : np.ndarray, optional
        Land/sea mask array (1=water, 0=land)
        
    Returns:
    --------
    W : np.ndarray
        Linear interpolation weights (4 x npoints)
        Weight order: SW, SE, NE, NW (counter-clockwise from southwest)
        
    Examples:
    ---------
    >>> # Example contact points
    >>> xrg = np.array([1.5, 2.3, 3.7])  # Fractional XI coordinates
    >>> erg = np.array([2.1, 3.4, 1.9])  # Fractional ETA coordinates  
    >>> idg = np.array([1, 2, 3])        # Integer I indices
    >>> jdg = np.array([2, 3, 1])        # Integer J indices
    >>> weights = linear_weights(xrg, erg, idg, jdg)
    >>> print(f"Weights shape: {weights.shape}")  # (4, 3)
    >>> print(f"Weights sum: {np.sum(weights, axis=0)}")  # Should be ~[1, 1, 1]
    
    Notes:
    ------
    The interpolation weights are computed using bilinear interpolation:
    - W[0] = (1-p)*(1-q)  # Southwest corner weight
    - W[1] = p*(1-q)      # Southeast corner weight  
    - W[2] = p*q          # Northeast corner weight
    - W[3] = (1-p)*q      # Northwest corner weight
    
    where p = xrg - idg and q = erg - jdg are the fractional distances
    within the grid cell.
    """
    
    npoints = len(xrg)
    W = np.zeros((4, npoints))
    
    # Compute fractional distances within grid cells
    p = xrg - idg  # Fractional distance in XI direction
    q = erg - jdg  # Fractional distance in ETA direction
    
    # Handle NaN values (coincident points)
    p = np.where(np.isnan(p), 0, p)
    q = np.where(np.isnan(q), 0, q)
    
    # Bilinear interpolation weights
    # Weight order: (Idg,Jdg), (Idg+1,Jdg), (Idg+1,Jdg+1), (Idg,Jdg+1)
    W[0, :] = (1 - p) * (1 - q)  # Southwest corner
    W[1, :] = p * (1 - q)        # Southeast corner
    W[2, :] = p * q              # Northeast corner
    W[3, :] = (1 - p) * q        # Northwest corner
    
    # Apply land/sea masking if requested
    if impose_mask and mask is not None:
        for n in range(npoints):
            i, j = int(idg[n]), int(jdg[n])
            
            # Check bounds
            if i >= 0 and j >= 0 and i < mask.shape[0]-1 and j < mask.shape[1]-1:
                # Get mask values for 4-point stencil
                mask_weights = np.array([
                    mask[i, j],       # Southwest
                    mask[i+1, j],     # Southeast
                    mask[i+1, j+1],   # Northeast  
                    mask[i, j+1]      # Northwest
                ])
                
                # Apply mask to weights (zero weight for land points)
                W[:, n] *= mask_weights
                
                # Renormalize weights to sum to 1
                weight_sum = np.sum(W[:, n])
                if weight_sum > 0:
                    W[:, n] /= weight_sum
                else:
                    # All surrounding points are land - set weights to zero
                    W[:, n] = 0
    
    # Clean up numerical noise
    W[np.abs(W) < 100 * np.finfo(float).eps] = 0
    W[np.abs(W - 1) < 100 * np.finfo(float).eps] = 1
    
    return W


def quadratic_weights(xrg: np.ndarray, erg: np.ndarray,
                     idg: np.ndarray, jdg: np.ndarray,
                     impose_mask: bool = False,
                     mask: Optional[np.ndarray] = None,
                     alpha: float = 0.0) -> np.ndarray:
    """
    Compute quadratic horizontal interpolation weights for contact points.
    
    This function computes biquadratic interpolation weights for interpolating
    from donor grid points to receiver grid contact points. The weights are
    computed for a 9-point stencil around each contact point.
    
    Parameters:
    -----------
    xrg : np.ndarray
        Receiver grid XI-coordinates (fractional grid indices)
    erg : np.ndarray
        Receiver grid ETA-coordinates (fractional grid indices)
    idg : np.ndarray
        Donor grid I-indices (integer grid indices)
    jdg : np.ndarray
        Donor grid J-indices (integer grid indices)
    impose_mask : bool, optional
        Whether to impose land/sea masking (default False)
    mask : np.ndarray, optional
        Land/sea mask array (1=water, 0=land)
    alpha : float, optional
        Quadratic interpolation parameter (default 0.0)
        Controls the shape of the interpolation function
        
    Returns:
    --------
    W : np.ndarray
        Quadratic interpolation weights (9 x npoints)
        Weight arrangement in 3x3 stencil:
        6 7 8
        3 4 5  
        0 1 2
        
    Examples:
    ---------
    >>> # Example contact points
    >>> xrg = np.array([1.5, 2.3])  # Fractional XI coordinates
    >>> erg = np.array([2.1, 3.4])  # Fractional ETA coordinates
    >>> idg = np.array([1, 2])      # Integer I indices
    >>> jdg = np.array([2, 3])      # Integer J indices
    >>> weights = quadratic_weights(xrg, erg, idg, jdg)
    >>> print(f"Weights shape: {weights.shape}")  # (9, 2)
    >>> print(f"Weights sum: {np.sum(weights, axis=0)}")  # Should be ~[1, 1]
    
    Notes:
    ------
    The quadratic interpolation uses Lagrange basis functions:
    - Rm = 0.5*p*(p-1) + alpha     (for i-1 terms)
    - Ro = (1-p*p) - 2*alpha       (for i terms)  
    - Rp = 0.5*p*(p+1) + alpha     (for i+1 terms)
    
    And similarly for the ETA direction (Sm, So, Sp).
    The 9 weights are products of these basis functions.
    """
    
    npoints = len(xrg)
    W = np.zeros((9, npoints))
    
    # Compute fractional distances within grid cells
    p = xrg - idg  # Fractional distance in XI direction
    q = erg - jdg  # Fractional distance in ETA direction
    
    # Handle NaN values
    p = np.where(np.isnan(p), 0, p)
    q = np.where(np.isnan(q), 0, q)
    
    # Quadratic Lagrange basis functions in XI direction
    Rm = 0.5 * p * (p - 1) + alpha  # Weight for i-1 column
    Ro = (1 - p * p) - 2 * alpha    # Weight for i column
    Rp = 0.5 * p * (p + 1) + alpha  # Weight for i+1 column
    
    # Quadratic Lagrange basis functions in ETA direction
    Sm = 0.5 * q * (q - 1) + alpha  # Weight for j-1 row
    So = (1 - q * q) - 2 * alpha    # Weight for j row
    Sp = 0.5 * q * (q + 1) + alpha  # Weight for j+1 row
    
    # Compute 9-point stencil weights as tensor products
    # Weight arrangement:
    # 6 7 8    (i-1,j+1) (i,j+1) (i+1,j+1)
    # 3 4 5    (i-1,j)   (i,j)   (i+1,j)
    # 0 1 2    (i-1,j-1) (i,j-1) (i+1,j-1)
    W[0, :] = Rm * Sm  # (i-1, j-1)
    W[1, :] = Ro * Sm  # (i,   j-1)
    W[2, :] = Rp * Sm  # (i+1, j-1)
    W[3, :] = Rm * So  # (i-1, j)
    W[4, :] = Ro * So  # (i,   j)
    W[5, :] = Rp * So  # (i+1, j)
    W[6, :] = Rm * Sp  # (i-1, j+1)
    W[7, :] = Ro * Sp  # (i,   j+1)
    W[8, :] = Rp * Sp  # (i+1, j+1)
    
    # Apply land/sea masking if requested
    if impose_mask and mask is not None:
        for n in range(npoints):
            i, j = int(idg[n]), int(jdg[n])
            
            # Check bounds (need larger stencil for quadratic)
            if i >= 1 and j >= 1 and i < mask.shape[0]-1 and j < mask.shape[1]-1:
                # Get mask values for 9-point stencil
                mask_values = np.array([
                    mask[i-1, j-1], mask[i, j-1], mask[i+1, j-1],  # Bottom row
                    mask[i-1, j],   mask[i, j],   mask[i+1, j],    # Middle row
                    mask[i-1, j+1], mask[i, j+1], mask[i+1, j+1]   # Top row
                ])
                
                # Apply mask to weights
                W[:, n] *= mask_values
                
                # Check if any land points affect the interpolation
                mask_sum = np.sum(mask_values)
                if mask_sum < 9:  # At least one land point in stencil
                    weight_sum = np.sum(W[:, n])
                    if weight_sum > 0:
                        # Renormalize weights
                        W[:, n] /= weight_sum
                    else:
                        # All surrounding points are land
                        W[:, n] = 0
    
    # Clean up numerical noise
    W[np.abs(W) < 100 * np.finfo(float).eps] = 0
    W[np.abs(W - 1) < 100 * np.finfo(float).eps] = 1
    
    return W


def validate_weights(weights: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Validate interpolation weights.
    
    Checks that interpolation weights sum to unity (or close to it within
    numerical tolerance) for each point.
    
    Parameters:
    -----------
    weights : np.ndarray
        Interpolation weights array (nweights x npoints)
    tolerance : float, optional
        Numerical tolerance for weight sum check (default 1e-10)
        
    Returns:
    --------
    valid : bool
        True if all weights are valid, False otherwise
        
    Examples:
    ---------
    >>> weights = linear_weights(xrg, erg, idg, jdg)
    >>> is_valid = validate_weights(weights)
    >>> print(f"Weights are valid: {is_valid}")
    """
    weight_sums = np.sum(weights, axis=0)
    return np.allclose(weight_sums, 1.0, atol=tolerance)


def apply_weights(data: np.ndarray, weights: np.ndarray, 
                 stencil_indices: np.ndarray) -> np.ndarray:
    """
    Apply interpolation weights to data array.
    
    This is a utility function for applying pre-computed interpolation
    weights to interpolate data from donor grid to receiver grid points.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array on donor grid
    weights : np.ndarray
        Interpolation weights (nweights x npoints)
    stencil_indices : np.ndarray
        Grid indices for stencil points (nweights x npoints x 2)
        Last dimension contains [i, j] indices
        
    Returns:
    --------
    interpolated : np.ndarray
        Interpolated values at receiver grid points
        
    Examples:
    ---------
    >>> # Interpolate bathymetry to contact points
    >>> h_interp = apply_weights(donor_grid['h'], weights, stencil_indices)
    """
    nweights, npoints = weights.shape
    interpolated = np.zeros(npoints)
    
    for n in range(npoints):
        for w in range(nweights):
            if weights[w, n] != 0:  # Skip zero weights
                i, j = stencil_indices[w, n, :]
                if 0 <= i < data.shape[0] and 0 <= j < data.shape[1]:
                    interpolated[n] += weights[w, n] * data[i, j]
    
    return interpolated

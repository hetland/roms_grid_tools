"""
Data structures for ROMS nested grids.

This module contains the dataclasses and data structures used throughout
the ROMS grid tools package for representing grids, contact points,
and interpolation information.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ContactPoint:
    """
    Structure for contact point data between nested grids.
    
    This class holds all the contact point information including coordinates,
    grid indices, and interpolated field values at contact points.
    
    Attributes:
    -----------
    xrg_rho, erg_rho : np.ndarray
        Receiver grid fractional coordinates for RHO points
    Xrg_rho, Yrg_rho : np.ndarray
        Physical coordinates of RHO contact points
    Irg_rho, Jrg_rho : np.ndarray
        Receiver grid integer indices for RHO points
    Idg_rho, Jdg_rho : np.ndarray
        Donor grid integer indices for RHO points
        
    Similar arrays exist for U and V grid points.
    
    boundary_* : np.ndarray
        Boolean arrays indicating which points are on grid boundaries
        
    Field values at contact points (interpolated from donor grid):
    angle, f, h, pm, pn, dndx, dmde : np.ndarray
        Physical fields interpolated to contact points
    mask_* : np.ndarray
        Land/sea masks at contact points
        
    Examples:
    ---------
    >>> contact = ContactPoint()
    >>> contact.Irg_rho = np.array([10, 11, 12])
    >>> contact.Jrg_rho = np.array([20, 21, 22])
    >>> contact.h = np.array([100.5, 95.2, 88.7])  # Bathymetry
    """
    
    # RHO point contact data
    xrg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    erg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    Xrg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    Yrg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    Irg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    Jrg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    Idg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    Jdg_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # U point contact data
    xrg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    erg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    Xrg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    Yrg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    Irg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    Jrg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    Idg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    Jdg_u: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # V point contact data
    xrg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    erg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    Xrg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    Yrg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    Irg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    Jrg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    Idg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    Jdg_v: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Boundary flags for contact points
    boundary_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    boundary_u: np.ndarray = field(default_factory=lambda: np.array([]))
    boundary_v: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Donor grid field values interpolated to contact points
    angle: np.ndarray = field(default_factory=lambda: np.array([]))
    f: np.ndarray = field(default_factory=lambda: np.array([]))
    h: np.ndarray = field(default_factory=lambda: np.array([]))
    pm: np.ndarray = field(default_factory=lambda: np.array([]))
    pn: np.ndarray = field(default_factory=lambda: np.array([]))
    dndx: np.ndarray = field(default_factory=lambda: np.array([]))
    dmde: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Land/sea masks at contact points
    mask_rho: np.ndarray = field(default_factory=lambda: np.array([]))
    mask_u: np.ndarray = field(default_factory=lambda: np.array([]))
    mask_v: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def get_npoints(self, grid_type: str = 'rho') -> int:
        """
        Get number of contact points for specified grid type.
        
        Parameters:
        -----------
        grid_type : str
            Grid type ('rho', 'u', or 'v')
            
        Returns:
        --------
        npoints : int
            Number of contact points
        """
        if grid_type.lower() == 'rho':
            return len(self.Irg_rho)
        elif grid_type.lower() == 'u':
            return len(self.Irg_u)
        elif grid_type.lower() == 'v':
            return len(self.Irg_v)
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
    
    def get_total_points(self) -> int:
        """Get total number of contact points across all grid types."""
        return self.get_npoints('rho') + self.get_npoints('u') + self.get_npoints('v')


@dataclass 
class GridInfo:
    """
    Structure for grid information and dimensions.
    
    This class contains basic information about a ROMS grid including
    dimensions, refinement parameters, and index arrays.
    
    Attributes:
    -----------
    filename : str
        Grid NetCDF file name
    Lp, Mp : int
        Grid dimensions (including boundary points)
    L, M : int
        Interior grid dimensions
    refine_factor : int
        Grid refinement factor (for nested grids)
    parent_* : int
        Parent grid extraction region bounds
    I_*, J_* : np.ndarray
        Index arrays for different grid types (rho, psi, u, v)
        
    Examples:
    ---------
    >>> grid = GridInfo()
    >>> grid.filename = 'my_grid.nc'
    >>> grid.Lp, grid.Mp = 101, 81
    >>> grid.L, grid.M = 100, 80
    >>> print(f"Grid size: {grid.Lp} x {grid.Mp}")
    """
    
    filename: str = ""
    Lp: int = 0  # XI-direction RHO points (including boundaries)
    Mp: int = 0  # ETA-direction RHO points (including boundaries)
    L: int = 0   # XI-direction interior points (Lp - 1)
    M: int = 0   # ETA-direction interior points (Mp - 1)
    
    # Nested grid refinement parameters
    refine_factor: int = 1
    parent_Imin: int = 0
    parent_Imax: int = 0
    parent_Jmin: int = 0
    parent_Jmax: int = 0
    
    # Index arrays for different grid staggerings
    I_psi: Optional[np.ndarray] = None
    J_psi: Optional[np.ndarray] = None
    I_rho: Optional[np.ndarray] = None
    J_rho: Optional[np.ndarray] = None
    I_u: Optional[np.ndarray] = None
    J_u: Optional[np.ndarray] = None
    I_v: Optional[np.ndarray] = None
    J_v: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize derived quantities after object creation."""
        if self.L == 0 and self.Lp > 0:
            self.L = self.Lp - 1
        if self.M == 0 and self.Mp > 0:
            self.M = self.Mp - 1
    
    def create_index_arrays(self):
        """Create index arrays for all grid types."""
        if self.Lp > 0 and self.Mp > 0:
            # RHO points (includes boundaries)
            self.I_rho, self.J_rho = np.meshgrid(
                np.arange(1, self.Lp + 1), 
                np.arange(1, self.Mp + 1), 
                indexing='xy'
            )
            
            # PSI points (interior corners)
            self.I_psi, self.J_psi = np.meshgrid(
                np.arange(1, self.L + 1),
                np.arange(1, self.M + 1), 
                indexing='xy'
            )
            
            # U points (XI-edges)
            self.I_u, self.J_u = np.meshgrid(
                np.arange(1, self.L + 1),
                np.arange(1, self.Mp + 1), 
                indexing='xy'
            )
            
            # V points (ETA-edges)
            self.I_v, self.J_v = np.meshgrid(
                np.arange(1, self.Lp + 1),
                np.arange(1, self.M + 1), 
                indexing='xy'
            )
    
    def get_grid_spacing(self) -> float:
        """
        Estimate grid spacing (if uniform grid).
        
        Returns:
        --------
        spacing : float
            Estimated grid spacing in grid units
            
        Notes:
        ------
        This is a rough estimate assuming uniform spacing.
        For actual grid spacing, use the pm/pn metrics.
        """
        return 1.0 / self.refine_factor if self.refine_factor > 1 else 1.0


@dataclass
class ContactRegion:
    """
    Structure for contact region data between two grids.
    
    This class represents a contact region where two grids overlap
    or are connected, including the type of connection and boundary
    information.
    
    Attributes:
    -----------
    donor_grid, receiver_grid : int
        Grid numbers involved in this contact region (1-based indexing)
    coincident : bool
        True if grids have coincident boundaries
    composite : bool
        True if contact involves composite grids
    hybrid : bool
        True if contact involves hybrid vertical coordinates
    mosaic : bool
        True if contact involves mosaic grids
    refinement : bool
        True if contact involves grid refinement
    interior : Dict[str, Any]
        Interior contact region information
    corners : Dict[str, Any]
        Corner point information
    boundary : List[Dict[str, Any]]
        Boundary edge information (4 edges: W, S, E, N)
    point : ContactPoint
        Contact point data structure
        
    Examples:
    ---------
    >>> region = ContactRegion()
    >>> region.donor_grid = 1
    >>> region.receiver_grid = 2
    >>> region.refinement = True
    >>> region.boundary = [{'okay': False} for _ in range(4)]
    """
    
    donor_grid: int = 0
    receiver_grid: int = 0
    
    # Connection type flags
    coincident: bool = False
    composite: bool = False
    hybrid: bool = False
    mosaic: bool = False
    refinement: bool = False
    
    # Contact region geometry
    interior: Dict[str, Any] = field(default_factory=dict)
    corners: Dict[str, Any] = field(default_factory=dict)
    boundary: List[Dict[str, Any]] = field(default_factory=list)
    
    # Contact point data
    point: Optional[ContactPoint] = None
    
    def __post_init__(self):
        """Initialize default boundary structure."""
        if not self.boundary:
            # Initialize 4 boundary edges (Western, Southern, Eastern, Northern)
            self.boundary = [{'okay': False} for _ in range(4)]
    
    def get_connection_type(self) -> str:
        """
        Get string description of connection type.
        
        Returns:
        --------
        conn_type : str
            Description of connection type
        """
        if self.refinement:
            return "refinement"
        elif self.coincident:
            return "coincident"
        elif self.composite:
            return "composite"
        elif self.hybrid:
            return "hybrid"
        elif self.mosaic:
            return "mosaic"
        else:
            return "unknown"
    
    def has_active_boundaries(self) -> bool:
        """Check if any boundaries are active."""
        return any(boundary.get('okay', False) for boundary in self.boundary)
    
    def get_active_boundary_count(self) -> int:
        """Get number of active boundaries."""
        return sum(1 for boundary in self.boundary if boundary.get('okay', False))


@dataclass
class ContactStructure:
    """
    Main contact structure for nested grids.
    
    This is the top-level structure that contains all information
    about nested grids, their connections, and contact points.
    
    Attributes:
    -----------
    Ngrids : int
        Number of grids in the nested system
    Ncontact : int
        Number of contact regions
    NLweights : int
        Number of linear interpolation weights (4)
    NQweights : int
        Number of quadratic interpolation weights (9)
    Ndatum : int
        Total number of contact points
    spherical : bool
        True if using spherical coordinates
    
    Boundary edge constants:
    western_edge, southern_edge, eastern_edge, northern_edge : int
        Constants for boundary edge identification
        
    grid : List[GridInfo]
        Grid information for each grid
    contact : List[ContactRegion]
        Contact region information
    refined : List[Dict[str, Any]]
        Refined grid coordinate information
    Lweights : List[Dict[str, np.ndarray]]
        Linear interpolation weights
    Qweights : List[Dict[str, np.ndarray]]
        Quadratic interpolation weights
        
    Examples:
    ---------
    >>> S = ContactStructure()
    >>> S.Ngrids = 3
    >>> S.Ncontact = 4
    >>> S.spherical = True
    >>> print(f"Contact structure: {S.Ngrids} grids, {S.Ncontact} regions")
    """
    
    # Basic structure parameters
    Ngrids: int = 0
    Ncontact: int = 0
    NLweights: int = 4   # Linear interpolation uses 4 points
    NQweights: int = 9   # Quadratic interpolation uses 9 points
    Ndatum: int = 0      # Total number of contact points
    
    # Coordinate system
    spherical: bool = False
    
    # Boundary edge constants (1-based indexing to match MATLAB)
    western_edge: int = 1
    southern_edge: int = 2
    eastern_edge: int = 3
    northern_edge: int = 4
    
    # Grid and contact information
    grid: List[GridInfo] = field(default_factory=list)
    contact: List[ContactRegion] = field(default_factory=list)
    refined: List[Dict[str, Any]] = field(default_factory=list)
    
    # Interpolation weights
    Lweights: List[Dict[str, np.ndarray]] = field(default_factory=list)
    Qweights: List[Dict[str, np.ndarray]] = field(default_factory=list)
    
    def validate(self) -> bool:
        """
        Validate the contact structure consistency.
        
        Returns:
        --------
        valid : bool
            True if structure is valid
        """
        # Check basic consistency
        if len(self.grid) != self.Ngrids:
            return False
        if len(self.contact) != self.Ncontact:
            return False
            
        # Check contact region references
        for contact_region in self.contact:
            if (contact_region.donor_grid < 1 or 
                contact_region.donor_grid > self.Ngrids or
                contact_region.receiver_grid < 1 or 
                contact_region.receiver_grid > self.Ngrids):
                return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the contact structure.
        
        Returns:
        --------
        summary : Dict[str, Any]
            Summary information
        """
        summary = {
            'ngrids': self.Ngrids,
            'ncontact': self.Ncontact,
            'ndatum': self.Ndatum,
            'spherical': self.spherical,
            'grid_files': [grid.filename for grid in self.grid],
            'contact_types': [region.get_connection_type() for region in self.contact]
        }
        
        # Count different connection types
        connection_counts = {}
        for region in self.contact:
            conn_type = region.get_connection_type()
            connection_counts[conn_type] = connection_counts.get(conn_type, 0) + 1
        summary['connection_counts'] = connection_counts
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of the contact structure."""
        summary = self.get_summary()
        
        print("ROMS Nested Grids Contact Structure Summary")
        print("=" * 45)
        print(f"Number of grids: {summary['ngrids']}")
        print(f"Number of contact regions: {summary['ncontact']}")
        print(f"Total contact points: {summary['ndatum']}")
        print(f"Coordinate system: {'Spherical' if summary['spherical'] else 'Cartesian'}")
        
        print("\nGrid files:")
        for i, filename in enumerate(summary['grid_files']):
            print(f"  {i+1}: {filename}")
        
        print("\nConnection types:")
        for conn_type, count in summary['connection_counts'].items():
            print(f"  {conn_type}: {count}")


# Utility functions for working with data structures

def create_empty_contact_structure(ngrids: int) -> ContactStructure:
    """
    Create an empty contact structure for a given number of grids.
    
    Parameters:
    -----------
    ngrids : int
        Number of grids in the nested system
        
    Returns:
    --------
    S : ContactStructure
        Initialized empty contact structure
    """
    S = ContactStructure()
    S.Ngrids = ngrids
    S.Ncontact = (ngrids - 1) * 2 if ngrids > 1 else 0
    
    # Initialize empty grid and contact lists
    S.grid = [GridInfo() for _ in range(ngrids)]
    S.contact = [ContactRegion() for _ in range(S.Ncontact)]
    
    return S


def merge_contact_points(point1: ContactPoint, point2: ContactPoint) -> ContactPoint:
    """
    Merge two ContactPoint structures.
    
    Parameters:
    -----------
    point1, point2 : ContactPoint
        Contact point structures to merge
        
    Returns:
    --------
    merged : ContactPoint
        Merged contact point structure
    """
    merged = ContactPoint()
    
    # Merge RHO points
    merged.Irg_rho = np.concatenate([point1.Irg_rho, point2.Irg_rho])
    merged.Jrg_rho = np.concatenate([point1.Jrg_rho, point2.Jrg_rho])
    merged.Xrg_rho = np.concatenate([point1.Xrg_rho, point2.Xrg_rho])
    merged.Yrg_rho = np.concatenate([point1.Yrg_rho, point2.Yrg_rho])
    merged.Idg_rho = np.concatenate([point1.Idg_rho, point2.Idg_rho])
    merged.Jdg_rho = np.concatenate([point1.Jdg_rho, point2.Jdg_rho])
    
    # Merge U points
    merged.Irg_u = np.concatenate([point1.Irg_u, point2.Irg_u])
    merged.Jrg_u = np.concatenate([point1.Jrg_u, point2.Jrg_u])
    merged.Xrg_u = np.concatenate([point1.Xrg_u, point2.Xrg_u])
    merged.Yrg_u = np.concatenate([point1.Yrg_u, point2.Yrg_u])
    merged.Idg_u = np.concatenate([point1.Idg_u, point2.Idg_u])
    merged.Jdg_u = np.concatenate([point1.Jdg_u, point2.Jdg_u])
    
    # Merge V points
    merged.Irg_v = np.concatenate([point1.Irg_v, point2.Irg_v])
    merged.Jrg_v = np.concatenate([point1.Jrg_v, point2.Jrg_v])
    merged.Xrg_v = np.concatenate([point1.Xrg_v, point2.Xrg_v])
    merged.Yrg_v = np.concatenate([point1.Yrg_v, point2.Yrg_v])
    merged.Idg_v = np.concatenate([point1.Idg_v, point2.Idg_v])
    merged.Jdg_v = np.concatenate([point1.Jdg_v, point2.Jdg_v])
    
    # Merge field data
    if len(point1.h) > 0 and len(point2.h) > 0:
        merged.h = np.concatenate([point1.h, point2.h])
    elif len(point1.h) > 0:
        merged.h = point1.h.copy()
    elif len(point2.h) > 0:
        merged.h = point2.h.copy()
    
    # Similar for other fields...
    # (Implementation would continue for all other fields)
    
    return merged

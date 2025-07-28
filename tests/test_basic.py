"""
Test basic package functionality
"""

import numpy as np


def test_imports():
    """Test that basic imports work"""
    try:
        import roms_grid_tools as rgt
        assert hasattr(rgt, '__version__')
        print(f"✓ Package version: {rgt.__version__}")
        
        # Test individual module imports
        from roms_grid_tools.data_structures import ContactStructure, ContactPoint
        from roms_grid_tools.weights import linear_weights, quadratic_weights
        
        print("✓ All imports successful")
        
    except ImportError as e:
        raise ImportError(f"Import failed: {e}")


def test_data_structures():
    """Test data structure creation"""
    from roms_grid_tools.data_structures import ContactStructure, ContactPoint
    
    # Test ContactStructure
    S = ContactStructure()
    S.Ngrids = 2
    S.Ncontact = 2
    S.spherical = True
    
    assert S.Ngrids == 2
    assert S.Ncontact == 2
    assert S.spherical is True
    
    # Test ContactPoint
    contact = ContactPoint()
    contact.Irg_rho = np.array([1, 2, 3])
    contact.Jrg_rho = np.array([4, 5, 6])
    
    assert len(contact.Irg_rho) == 3
    assert contact.get_npoints('rho') == 3
    
    print("✓ Data structures work correctly")


def test_weights():
    """Test interpolation weight calculations"""
    from roms_grid_tools.weights import linear_weights, quadratic_weights, validate_weights
    
    # Test data
    xrg = np.array([1.5, 2.3, 3.7])
    erg = np.array([2.1, 3.4, 1.9])
    idg = np.array([1, 2, 3])
    jdg = np.array([2, 3, 1])
    
    # Test linear weights
    linear_w = linear_weights(xrg, erg, idg, jdg)
    assert linear_w.shape == (4, 3)
    assert validate_weights(linear_w)
    
    # Test quadratic weights
    quad_w = quadratic_weights(xrg, erg, idg, jdg)
    assert quad_w.shape == (9, 3)
    assert validate_weights(quad_w)
    
    print("✓ Weight calculations work correctly")


def test_grid_utils():
    """Test grid utility functions"""
    from roms_grid_tools.grid_utils import gcircle, inpolygon_numpy
    
    # Test great circle distance
    dist = gcircle(-122.0, 37.0, -121.0, 38.0)
    assert isinstance(dist, (float, np.ndarray))
    assert dist > 0
    
    # Test polygon function
    # Simple square
    xv = np.array([0, 1, 1, 0, 0])
    yv = np.array([0, 0, 1, 1, 0])
    xq = np.array([0.5, 1.5])
    yq = np.array([0.5, 0.5])
    
    inside = inpolygon_numpy(xq, yq, xv, yv)
    assert inside[0] is True   # Point inside
    assert inside[1] is False  # Point outside
    
    print("✓ Grid utilities work correctly")


if __name__ == "__main__":
    test_imports()
    test_data_structures()
    test_weights()
    test_grid_utils()
    print("\n✓ All tests passed!")

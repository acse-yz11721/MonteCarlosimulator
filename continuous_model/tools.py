"""
This module contains a lots of helper fucntion such
as, the function to update spin, , etc.
"""
# I decided to implement two ways of generating random unit vectors
# 1. uniform distribution of points on the surface of a sphere
# 2. renew old spin with epsilon = 1e-3
import numpy as np




def init_vector():
    """
    Generating random unit vectors that can fairly represent all directions in an 
    3-dimensional space.
    
    Returns
    -------
    numpy.ndarray
        the unit vector in an 3-dimensional space (as an (3,) array)
        
    """

    theta = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(0, 1)
    phi = np.arccos(2 * u - 1)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.array([x, y, z])


def spin_rotation(unit_vector):
    """
    Creates an new unit vector of size (3, ) by using the previous state of 
    unit vecor.
    
    Parameters
    ----------
    unit_vector : (3,) numpy.ndarray
        old state of spin in discretised magnetisation Field `m`
    
    Returns
    -------
    numpy.ndarray
        the unit vector in an 3-dimensional space (as an (3,) array)
    """
    
    e = 1e-3 * np.random.uniform(-1, 1, 3)
    unit_vector = unit_vector + e
    updated_unit_vector = unit_vector / np.linalg.norm(unit_vector)

    return updated_unit_vector


def randomMagnetisation_field(mesh):
    """
    Create a random normalised magnetisation Field `m`, in which 
    we assume the size of the Field is (nx, ny, nz).
    
    Parameters
    ----------
    mesh : continuous_model.CuboidMesh
        Finite-difference mesh. In which we assume the number of cells in
        the mesh is n_x * n_y * n_z.
    Returns:
    -------
    numpy.ndarray
        numpy array of size (nx, ny, nz, 3) which represents a discretised 
        magnetisation Field `m`.
        
    """
    cells_x = mesh.nx
    cells_y = mesh.ny
    cells_z = mesh.nz
    
    m_init = np.random.uniform(-1, 1, (cells_x, cells_y, cells_z, 3))
    m_norm = np.linalg.norm(m_init, axis=3)
    m_norm = np.expand_dims(m_norm, axis=3)
    m = m_init / m_norm
    return m

def scalarMagnetisation_field(mesh, value):
    """
    Create a scalar normalised magnetisation Field `m`, in which 
    we assume the size of the Field is (nx, ny, nz).
    
    Parameters
    ----------
    mesh : continuous_model.CuboidMesh
        Finite-difference mesh. In which we assume the number of cells in
        the mesh is n_x * n_y * n_z.
    Returns:
    -------
    numpy.ndarray
        numpy array of size (nx, ny, nz, 3) which represents a discretised 
        magnetisation Field `m`.
        
    """
    
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise TypeError("value must be numpy.ndarray, list or list")
    
    if np.array(value).size != 3:
        raise ValueError('dimension mismatch')
    
    value = np.array(value)
    
    cells_x = mesh.nx
    cells_y = mesh.ny
    cells_z = mesh.nz
    
    m = np.tile(value, cells_x * cells_y * cells_z)
    m = m.reshape((cells_x, cells_y, cells_z, 3))
      
    return m
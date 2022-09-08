"""Module constructing a Finite-difference field """


import numpy as np
import math
from textwrap import dedent




class m_Field:
    def __init__(self, mesh, Ms, m):
        """
        This class represents a Finite-difference field, in which we defines 
        several operations to manipulate normalised magnetisation field m.

        Parameters
        ----------
        mesh : continuous_model.RectangularMesh
            Finite-difference mesh. In which we assume the number of cells in
            the mesh is n_x * n_y * n_z.
        
        Ms : float
            magnetisation saturation  

        m : array_like
            normalised magnetisation field, norm = 1

        """

        self._mu0 = 4 * np.pi * 1e-7  # Fundamental constants
        self.a_x = mesh.units * mesh.dx
        self.a_y = mesh.units * mesh.dy
        self.a_z = mesh.units * mesh.dz
        self.nx = mesh.nx
        self.ny = mesh.ny
        self.nz = mesh.nz
        self.v = self.a_x * self.a_y * self.a_z
        self.Ms = Ms

        if isinstance(m, (np.ndarray, list, tuple)):
            self.m = np.array(m)
            if self.m.shape != (self.nx, self.ny, self.nz, 3):
                raise ValueError("Dimension mismatch, please enter a new m to match the mesh")

        else:
            raise TypeError("m must be numpy.ndarray, list or list")

        # if radius:        
        #     origin_point = [self.a_x * self.nx / 2, self.a_y * self.ny / 2]
        #     for i in range(self.nx):
        #         for j in range(self.ny):
        #             for k in range(self.nz):
        #                 point = [(i+1/2) * self.a_x, (j+1/2) * self.a_y]
        #                 if math.dist(origin_point, point) > radius:
        #                     self.m[i, j, k] = [0, 0, 0]

    @property
    def array(self):
        return self.m

    def curl(self):
        """
        Computes the curl operator of discretised magnetisation Field `m` with grid spacing
        `a_x` in x direction, `a_y` in y diection and `a_z` in z direction. Assuming Dirichlet
        Boundary conditions.
        
        Returns
        -------

        curl_operator : ndarray
            Curl of tensor field `m`.
             
        """
        curl_operator = np.zeros(self.array.shape)
        
        # I tried Dirichlet boundary condition this time, it works I can have the same result as ubermag
        # assign values to the boundary points and inner points 
        m_padding = np.zeros((self.array.shape[0]+2, self.array.shape[1]+2, self.array.shape[2]+2, 3))
        # reassign values to inner points
        m_padding[1:-1, 1:-1, 1:-1] = self.array[:, :, :]

        
        # central difference for the inner points
        dFy_dx = 1 / (2 * self.a_x) * (m_padding[2:, 1:-1, 1:-1, 1] -m_padding[:-2, 1:-1, 1:-1, 1])
        dFz_dx = 1 / (2 * self.a_x) * (m_padding[2:, 1:-1, 1:-1, 2] -m_padding[:-2, 1:-1, 1:-1, 2])
        
        dFx_dy = 1 / (2 * self.a_y) * (m_padding[1:-1, 2:, 1:-1, 0] -m_padding[1:-1, :-2, 1:-1, 0]) 
        dFz_dy = 1 / (2 * self.a_y) * (m_padding[1:-1, 2:, 1:-1, 2] -m_padding[1:-1, :-2, 1:-1, 2]) 

        dFx_dz = 1 / (2 * self.a_z) * (m_padding[1:-1, 1:-1, 2:, 0] -m_padding[1:-1, 1:-1, :-2, 0])
        dFy_dz = 1 / (2 * self.a_z) * (m_padding[1:-1, 1:-1, 2:, 1] -m_padding[1:-1, 1:-1, :-2, 1])
        
        curl_operator[..., 0] = dFz_dy - dFy_dz
        curl_operator[..., 1] = dFx_dz - dFz_dx
        curl_operator[..., 2] = dFy_dx - dFx_dy
        
        return curl_operator, (dFy_dx, dFz_dx, dFx_dy, dFz_dy, dFx_dz, dFy_dz), m_padding
    
    def curl_iter(self, spin_loc, old_curl_component, old_m_padding):
        """
        Similar to curl function above, this function is also used to compute
        the curl operator of discretised magnetisation Field `m`. But this function 
        accelerate the speed of monte carlo simulation as I only renew the line curl
        for x, y, z axis each time.
        
        Parameters
        ----------
        spin_loc: ndarray
            the selected coordinates of the spin
            
        old_curl_component: tuple
            contains the components of the curl operator in the previous state: 
            curl in x direction , curl in y direction and curl in z direction.
        
        old_m_padding: ndarray
            the extension of 'm' in the previous state.

        Returns
        -------

        curl_component: tuple
            contains the components of the curl operator in the current state: 
            curl in x direction , curl in y direction and curl in z direction.
        
        m_padding: ndarray
            the extension of 'm' in the current state.

        """
        x, y, z = spin_loc[0], spin_loc[1], spin_loc[2]
        dFy_dx = old_curl_component[0].copy()
        dFz_dx = old_curl_component[1].copy()
        dFx_dy = old_curl_component[2].copy()
        dFz_dy = old_curl_component[3].copy()
        dFx_dz = old_curl_component[4].copy()
        dFy_dz = old_curl_component[5].copy()

        m_padding = old_m_padding.copy()
        m_padding[x+1, y+1, z+1] = self.array[x, y, z]
        dFy_dx[:, y, z] = 1 / (2 * self.a_x) * (m_padding[2:, y+1, z+1, 1] - m_padding[:-2, y+1, z+1, 1])
        dFz_dx[:, y, z] = 1 / (2 * self.a_x) * (m_padding[2:, y+1, z+1, 2] - m_padding[:-2, y+1, z+1, 2])

        dFx_dy[x, :, z] = 1 / (2 * self.a_y) * (m_padding[x+1, 2:, z+1, 0] - m_padding[x+1, :-2, z+1, 0])
        dFz_dy[x, :, z] = 1 / (2 * self.a_y) * (m_padding[x+1, 2:, z+1, 2] - m_padding[x+1, :-2, z+1, 2])

        dFx_dz[x, y, :] = 1 / (2 * self.a_z) * (m_padding[x+1, y+1, 2:, 0] - m_padding[x+1, y+1, :-2, 0])
        dFy_dz[x, y, :] = 1 / (2 * self.a_z) * (m_padding[x+1, y+1, 2:, 1] - m_padding[x+1, y+1, :-2, 1])

        return (dFy_dx, dFz_dx, dFx_dy, dFz_dy, dFx_dz, dFy_dz), m_padding

    def laplace(self):
        """
        Computes the laplace operator of discretised magnetisation Field `m` with grid spacing
        `a_x` in x direction, `a_y` in y diection and `a_z` in z direction. Assuming Neumnn
        Boundary conditions.

        Returns
        -------

        laplace_operator : ndarray
            laplace operator of tensor discretised magnetisation Field `m`. 

        """
    
        laplace_operator = np.zeros(self.array.shape)

        # I tried Neumnn boundary condition this time, it works I can have the same result as ubermag
    
        m_padding = np.pad(self.array, ((1, 1), (1, 1), (1, 1), (0, 0)), 'edge')
        # central difference for the inner points
        # x direction
        laplace_operator_x = (1 / self.a_x**2) * (m_padding[2:, 1:-1, 1:-1] - 
                2 * m_padding[1:-1, 1:-1, 1:-1] + m_padding[:-2, 1:-1, 1:-1])
        # y direction
        laplace_operator_y = (1 / self.a_y**2) * (m_padding[1:-1, 2:, 1:-1] - 
                2 * m_padding[1:-1, 1:-1, 1:-1] + m_padding[1:-1, :-2, 1:-1])
        # z direction
        laplace_operator_z = (1 / self.a_z**2) * (m_padding[1:-1, 1:-1, 2:] -
                2 * m_padding[1:-1, 1:-1, 1:-1] + m_padding[1:-1, 1:-1, :-2])
        laplace_operator = laplace_operator_x + laplace_operator_y + laplace_operator_z

        return laplace_operator, (laplace_operator_x, laplace_operator_y, laplace_operator_z), m_padding

    def laplace_iter(self, spin_loc, old_laplace_component, old_m_padding):
        """
        Similar to laplace function above, this function is also used to compute
        the laplace operator ofdiscretised magnetisation Field `m`. But this function 
        accelerate the speed of monte carlo simulation as I only renew the line laplace 
        for x, y, z axis each time.
        
        Parameters
        ----------
        spin_loc: ndarray
            the selected coordinates of the spin
            
        old_laplace_component: tuple
            contains the components of the laplace operator in the previous state: 
            laplace_operator_x, laplace_operator_y and laplace_operator_z.
        
        old_m_padding: ndarray
            the extension of 'm' in the previous state.

        Returns
        -------

        laplace_component: tuple
            contains the components of the laplace operator in the current state:
            laplace_operator_x, laplace_operator_y and laplace_operator_z.
        
        old_m_padding: ndarray
            the extension of 'm' in the current state.

        """
        x, y, z = spin_loc[0], spin_loc[1], spin_loc[2]

        
        new_laplace_operator_x = old_laplace_component[0].copy()
        new_laplace_operator_y = old_laplace_component[1].copy()
        new_laplace_operator_z = old_laplace_component[2].copy()

        if (x == (self.nx-1) or 0) or (y == (self.ny-1) or 0) or (z == (self.nz-1) or 0):
            m_padding = np.pad(self.array, ((1, 1), (1, 1), (1, 1), (0, 0)), 'edge')

        else:
            m_padding = old_m_padding.copy()
            m_padding[x+1, y+1, z+1] = self.array[x, y, z]
        # Instead of renew exchange for the while Field
        # Only renew the line energy for x, y, z axis each time
        new_laplace_operator_x[:, y, z] = (1 / self.a_x ** 2) * (
                m_padding[2:, y+1, z+1] - 2 * m_padding[1:-1, y+1, z+1] + m_padding[:-2, y+1, z+1])
        
        new_laplace_operator_y[x, :, z] = (1 / self.a_y ** 2) * (
                m_padding[x+1, 2:, z+1] - 2 * m_padding[x+1, 1:-1, z+1] + m_padding[x+1, :-2, z+1])

        new_laplace_operator_z[x, y, :] = (1 / self.a_z ** 2) * (
                m_padding[x+1, y+1, 2:] - 2 * m_padding[x+1, y+1, 1:-1] + m_padding[x+1, y+1, :-2])

        return (new_laplace_operator_x, new_laplace_operator_y, new_laplace_operator_z), m_padding

    def __repr__(self):       
        fmt = dedent("""\
        Cuboid Field
        Magnetitude of m(Ms): {:.2f}
        Cell size: ({}, {}, {})
    
        """)

        return fmt.format(self.Ms, self.a_x, self.a_y, self.a_z)
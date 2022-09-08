"""
This module is used to create a mesh
"""
# I was enlightened to write RectangularMesh class from Fidimag
# https://github.com/computationalmodelling/fidimag/blob/master/fidimag/common/cuboid_mesh.py
import numpy as np
from textwrap import dedent

class RectangularMesh:
    
    def __init__(self, dx=1, dy=1, dz=1, nx=1, ny=1, nz=1, units=5e-9):
        """
        Create Finite-difference mesh. In which we assume the number of cells in
        the mesh is n_x * n_y * n_z. The length of each cell in x dirction is
        dx * units, in y dirction is dy * units, in z direction is dy * units.
        
        Parameters
        ----------
        dx : float, optional
            length of cell in x direction, dimensionless
            
        dy : float, optional
            length of cell in y direction, dimensionless
            
        dz : float, optional
            length of cell in z direction, dimensionless
            
        nx :  int, optional
            number of cells in x direction in mesh
            
        ny :  int, optional
            number of cells in y direction in mesh
            
        nz :  int, optional
            number of cells in z direction in mesh
            
        units : float, optional
            assumed to be the unit for dx, dy and dz. Defaults to 5e-9.
            
        """
        
        if not all(isinstance(i, int) for i in [nx, ny, nz]):
             raise TypeError("nx, ny and nz must be integers")
            
        if (np.array([nx, ny, nz]) <= 0).any():
            raise ValueError("nx, ny and nz must be integers >= 1")
        
        if (np.array([dx, dy, dz]) <= 0).any():
            raise ValueError("dx, dy and dz must be greater than 0")

        self.dx = dx  
        self.dy = dy  
        self.dz = dz  
        self.nx = nx
        self.ny = ny  
        self.nz = nz
        self.units = units
        
        self.Lx = self.dx * self.nx * self.units   # length of mesh in x direction
        self.Ly = self.dy * self.ny * self.units   # length of mesh in y direction
        self.Lz = self.dz * self.nz * self.units   # length of mesh in z direction
        self.n = nx * ny * nz  # total number of cells
        
        self.mesh_size = (self.Lx, self.Ly, self.Lz)
        
        self.cell_size = (self.dx * self.units, \
                          self.dy * self.units, \
                          self.dz * self.units)
        

    def cells_index(self):
        """
        Create a generator to traverse all the cell indices.
        """
        cells = np.zeros([self.nx, self.ny, self.nz])
        for idx, _ in np.ndenumerate(cells):
            yield idx
        
        
    def __repr__(self):
        fmt = dedent("""
        Cuboid Mesh
        
        Total number of cells = {}
        Mesh size : {}
        Cell size : {}
        
        """)
        return fmt.format(self.n, self.mesh_size, self.cell_size)


if __name__ == '__main__':
    my_cool_mesh = RectangularMesh(1, 1, 1, 2, 2, 2)
    print(my_cool_mesh)


    # test generator
    it = my_cool_mesh.cells_index()
    for x in it:
        print(x)
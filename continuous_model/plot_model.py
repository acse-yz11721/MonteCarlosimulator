"""
This module was create to plot the micromagnetic model
please note that I did not write plot function by myself.
Instead I used the existing python package named ubermag.
"""
import numpy as np
import discretisedfield as df
import micromagneticmodel as mm

import discretisedfield.plotting as dfp


def plot_mesh(mesh, mode=0):
    """
    Call ubermag to plot the mesh using k3d.voxels if mode = 0,
    in addition, plot the mesh using matplotlib if mode = 1
    
    Parameters
    ----------
    mesh : continuous_model.RectangularMesh
        finite-difference mesh. In which we assume the number of cells in
        the mesh is n_x * n_y * n_z.
    mode: int
        if mode = 0 using k3d.voxels, if mode = 1 using matplotlib
    
    """
    
    u_mesh_size = mesh.mesh_size
    u_cell_size = mesh.cell_size
    region = df.Region(p1=(0, 0, 0), p2=u_mesh_size)
    mesh = df.Mesh(region=region, cell=u_cell_size)
    if not mode:
        mesh.k3d()
    else:
        mesh.mpl()


def plot_field(mesh, m_field, axis, value=None, save_path=None, save_name=None):
    """
    Call ubermag to plot the normalised magnetisation field
    
    Parameters
    ----------
    mesh : continuous_model.RectangularMesh
        finite-difference mesh. In which we assume the number of cells in
        the mesh is n_x * n_y * n_z.
        
    m_field : continuous_model.m_field
        represents a magnetisation field
    
    axis: str
        field can be sliced with a plane 'x', 'y' or 'z'
    
    save_path: str, optional
        the plot will be save in the defined path
    
    save_name: str, optional
        we name after the plot
    
    
    """
    # print(mesh.Ly)
    u_mesh_size = mesh.mesh_size
    u_cell_size = mesh.cell_size
    x_1, x_2 =  -u_mesh_size[0] / 2, u_mesh_size[0] / 2
    y_1, y_2 =  -u_mesh_size[1] / 2, u_mesh_size[1] / 2
    z_2 = u_mesh_size[2]
    region = df.Region(p1=(x_1, y_1, 0), p2=(x_2, y_2, z_2))
    u_mesh = df.Mesh(region=region, cell=u_cell_size)
    
    u_field = df.Field(u_mesh, dim=3, value=m_field.array, norm=m_field.Ms)

    if value == None:
        if axis == 'x':
            u_field.plane('x').mpl()
        elif axis == 'y':
            u_field.plane('y').mpl()
        elif axis =='z':
            u_field.plane('z').mpl()
        else:
            raise Exception('Enter a valid axis from x, y, z')
    else:
        if axis == 'x' and value >=0 and value <= mesh.Lx / 2:
            u_field.plane(x=value).mpl()
        elif axis == 'y' and value >=0 and value <= mesh.Ly / 2:
            u_field.plane(y=value).mpl()
        elif axis =='z'and value >=0 and value <= mesh.Lz:
            u_field.plane(z=value).mpl()
        else:
            raise Exception('Check axis or value, seems something went wrong')
    
    if save_path and save_name:
        u_field.write(save_path + save_name + ".vtk")
        

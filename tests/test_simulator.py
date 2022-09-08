import numpy as np
import os
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
import pytest
import continuous_model as cm  # my own package


# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly
@pytest.fixture(scope='module')
def m_1():
    # manipulate directories
    BASE_PATH = os.path.dirname(__file__)

    value = np.load(os.sep.join((BASE_PATH,'/initial_state/m_uber_8.npy')))
    return value

@pytest.fixture(scope='module')
def m_2():
    BASE_PATH = os.path.dirname(__file__)
    value = np.load(os.sep.join((BASE_PATH,'initial_state/mc_uber_2.npy')))
    return value

@pytest.fixture(scope='module')
def m_3():
    BASE_PATH = os.path.dirname(__file__)
    value = np.load(os.sep.join((BASE_PATH,'initial_state/mc_uber_1.npy')))
    return value
    
@pytest.fixture(scope='module')
def mesh_1():
    my_cool_mesh = cm.RectangularMesh(nx = 1, ny = 3, nz = 1, units = 5e-9)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_1_cuboid():
    my_cool_mesh = cm.RectangularMesh(dx = 2, dy= 2, dz= 1, nx = 1, ny = 3, nz = 1, units = 5e-9)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_2():
    my_cool_mesh = cm.RectangularMesh(nx = 2, ny = 1, nz = 3, units = 5e-9)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_2_cuboid():
    my_cool_mesh = cm.RectangularMesh(dx = 3, dy = 2, dz = 3, nx = 2, ny = 1, nz = 3, units = 5e-9)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_3():
    my_cool_mesh = cm.RectangularMesh(nx = 2, ny = 2, nz = 2, units = 5e-9)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_3_cuboid():
    my_cool_mesh = cm.RectangularMesh(dx=1, dy=2, dz=1, nx = 2, ny = 2, nz = 2, units = 5e-9)
    return my_cool_mesh

def is_parallel(m, x, y, z):
    flag = True
    neighbours = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    for neighbour in neighbours:
        dx = neighbour[0]
        dy = neighbour[1]
        dz = neighbour[2]
        if x+dx <0:
            continue
        if y+dy <0:
            continue
        if z+dz <0:
            continue
        if x+dx >= m.shape[0]:
            continue
        if y+dy >= m.shape[1]:
            continue
        if z+dz >= m.shape[2]:
            continue

        value = 1.0 - abs(np.dot(m[x][y][z], m[x+dx][y+dy][z+dz]))
        print(value)
        if not np.isclose(value, 0, atol=1e-3):
             flag = False
    return flag

def is_perpendicular(m, x, y, z):
    flag = True
    neighbours = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    for neighbour in neighbours:
        dx = neighbour[0]
        dy = neighbour[1]
        dz = neighbour[2]
        if x+dx <0:
            continue
        if y+dy <0:
            continue
        if z+dz <0:
            continue
        if x+dx >= m.shape[0]:
            continue
        if y+dy >= m.shape[1]:
            continue
        if z+dz >= m.shape[2]:
            continue

        value = abs(np.dot(m[x][y][z], m[x+dx][y+dy][z+dz]))
        print(value)
        if not np.isclose(value, 0, atol=5e-3):
            flag = False
    return flag

def is_parallel_H(m, x, y, z, H):
    flag = True
    
    value = 1.0 - abs(np.dot(m[x][y][z], H / np.linalg.norm(H)))
    print(value)
    if not np.isclose(value, 0, atol=1e-3):
             flag = False
    return flag

class TestMonteCarloTerms(object):
    @pytest.mark.parametrize("mesh, m, A, Ms", [
        ("mesh_1", "m_1", 1e-12, 8e5), 
        ("mesh_2", "m_2",  1.3e-11, 3.84e5),
        ("mesh_3", "m_3", 8.78e-12, 1.1e6)])
    def test_mc_exchange(self, mesh, m, A, Ms, request):
        """
        Test the exchange energy
        """
        # my program --- energy
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a class instance of m_Field
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        my_cool_simulator = cm.Simulator(A, 0, 0, 0, 0, 0)
        m_final, _ = my_cool_simulator.compute_minimum(my_cool_m, 0.001, 200000)

        expected_flag, flag = True, True
        nx, ny, nz = my_cool_mesh.nx, my_cool_mesh.ny, my_cool_mesh.nz
        for i in range(nx):
            for j in range(ny):
                for k in range(nz): 
                    # exchange
                    if not is_parallel(m_final, i, j ,k):
                        flag = False
    
        assert (flag == expected_flag)  # magnetitude test

    @pytest.mark.parametrize("mesh, m, D, Ms", [
        ("mesh_1", "m_1", 1.58e-3, 8e5)])
    def test_mc_dmi(self, mesh, m, D, Ms, request):
        """
        Test the exchange energy
        """
        # my program --- energy
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a class instance of m_Field
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        my_cool_simulator = cm.Simulator(0, D, 0, 0, 0, 0)
        m_final, _ = my_cool_simulator.compute_minimum(my_cool_m, 0.001, 300000)

        expected_flag, flag = True, True
        nx, ny, nz = my_cool_mesh.nx, my_cool_mesh.ny, my_cool_mesh.nz
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # exchange
                    if not is_perpendicular(m_final, i, j, k):
                        flag = False
    
        assert (flag == expected_flag)  # magnetitude test

    @pytest.mark.parametrize("mesh, m, H, Ms", [
    ("mesh_1", "m_1", [0, 0, 0.1/(4 * np.pi * 1e-7)], 3.84e5),
    ("mesh_2", "m_2",  [0, 0.1/(4 * np.pi * 1e-7), 0], 3.84e5), 
    ("mesh_3", "m_3", [10*np.sqrt(3)/2, 0, 10*1/2], 1.1e6)])
    def test_mc_zeeman(self, mesh, m, H, Ms, request):
        """
        Test the zeeman energy
        """
        # my program --- energy
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a class instance of m_Field
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        my_cool_simulator = cm.Simulator(0, 0, H, 0, 0, 0)
        m_final, _ = my_cool_simulator.compute_minimum(my_cool_m, 0.001, 200000)

        expected_flag, flag = True, True
        nx, ny, nz = my_cool_mesh.nx, my_cool_mesh.ny, my_cool_mesh.nz
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # zeeman
                    if not is_parallel_H(m_final, i, j, k, H):
                        flag = False

        assert (flag == expected_flag)  # magnetitude test
    
    @pytest.mark.parametrize('mesh, m, K, u, Ms', [
    ("mesh_1", "m_1", 0.7e6, (np.sqrt(2)/2, 1/2, 1/2), 8e5), 
    ("mesh_2", "m_2", 0.2e6, (1, 0, 0), 8e5),
    ("mesh_3", "m_3", 0.2e6, (np.sqrt(3)/2, 0, 1/2), 8e5)])
    def test_mc_UniaxialAnisotropy(self, mesh, m, K, u, Ms, request):
        """
        Test the UniaxialAnisotropy energy
        """
         # my program --- energy
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a class instance of m_Field
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        my_cool_simulator = cm.Simulator(0, 0, 0, K, u, 0)
        m_final, _ = my_cool_simulator.compute_minimum(my_cool_m, 0.001, 200000)
        
        # ubermag
        mesh_size = my_cool_mesh.mesh_size
        cell_size = my_cool_mesh.cell_size
        region = df.Region(p1=(0, 0, 0), p2=mesh_size)
        mesh = df.Mesh(region=region, cell=cell_size)
        
        m_uber = df.Field(mesh, dim=3, value=m, norm=Ms)
        system = mm.System(name='montecarlo_anitrosopy')
        system.m = m_uber
     
        system.energy = mm.UniaxialAnisotropy(K=K, u=u)
        md = oc.MinDriver()
        md.drive(system)
        m_uber_final = system.m.array / Ms
        flag = np.allclose(m_final, m_uber_final, atol=0.01)
        expected_flag = True
        assert (flag == expected_flag)

    @pytest.mark.parametrize("mesh, m, A, D, H, K, u, Ms", [
    ("mesh_1", "m_1", 8.78e-12, 1.58e-3, [0, 0, 0.1/(4 * np.pi * 1e-7)], 
     0.7e6, (np.sqrt(2)/2, 1/2, 1/2), 3.84e5), 
    ("mesh_1_cuboid", "m_1", 8.78e-12, 1.58e-3, [0, 0, 0.1/(4 * np.pi * 1e-7)], 
     0.7e6, (np.sqrt(2)/2, 1/2, 1/2), 3.84e5), 
    ("mesh_2_cuboid", "m_2",  8.78e-12, 1.58e-3, [0, 0, 0.1/(4 * np.pi * 1e-7)], 
     0.2e6, (1, 0, 0), 3.84e5),
    ("mesh_3", "m_3", 8.78e-12, 1.58e-3, [0, 0, 0.1/(4 * np.pi * 1e-7)], 0.2e6,
     (np.sqrt(3)/2, 0, 1/2), 3.84e5),
    ("mesh_3_cuboid", "m_3", 8.78e-12, 1.58e-3, [0, 0, 0.1/(4 * np.pi * 1e-7)], 0.2e6,
     (np.sqrt(3)/2, 0, 1/2), 3.84e5)])
    def test_mc_all_terms(self, mesh, m, A, D, H, K, u, Ms, request):
        """
        Test the exchange energy
        """
        # my program --- energy
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a class instance of m_Field
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        my_cool_simulator = cm.Simulator(A, D, H, K, u, 0)
        m_final, _ = my_cool_simulator.compute_minimum(my_cool_m, 0.001, 300000)

        # ubermag
        mesh_size = my_cool_mesh.mesh_size
        cell_size = my_cool_mesh.cell_size
        region = df.Region(p1=(0, 0, 0), p2=mesh_size)
        mesh = df.Mesh(region=region, cell=cell_size)
        
        m_uber = df.Field(mesh, dim=3, value=m, norm=Ms)
        system = mm.System(name='montecarlo_all_terms')
        system.m = m_uber
        system.energy = (mm.Exchange(A=A) + 
                 mm.Zeeman(H=H) + 
                 mm.DMI(D=D, crystalclass='T') + 
                 mm.UniaxialAnisotropy(K=K, u=u)
                )
        md = oc.MinDriver()
        md.drive(system)
        m_uber_final = system.m.array / Ms
        flag = np.allclose(m_final, m_uber_final, atol=0.01)
        expected_flag = True
        assert (flag == expected_flag)

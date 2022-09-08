"""
This module contains the tests of the continuous_model module.
"""
import pytest
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
import numpy as np
import os
import continuous_model as cm  # my own package



# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly
@pytest.fixture(scope='module')
def m_1():
    # manipulate directories
    BASE_PATH = os.path.dirname(__file__)
    # parent_dir = os.path.dirname(BASE_PATH)
    
    value = np.load(os.sep.join((BASE_PATH,'initial_state/m_uber_10.npy')))
    return value

@pytest.fixture(scope='module')
def m_2():
    BASE_PATH = os.path.dirname(__file__)
    # parent_dir = os.path.dirname(BASE_PATH) 
    value = np.load(os.sep.join((BASE_PATH,'initial_state/m_uber_11.npy')))
    return value

@pytest.fixture(scope='module')
def mesh_1():
    my_cool_mesh = cm.RectangularMesh(nx = 8, ny = 16, nz = 4, units = 5e-9)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_1_cuboid():
    my_cool_mesh = cm.RectangularMesh(dx = 2, dy = 4, dz = 1, nx = 8, ny = 16, nz = 4, units = 5e-9)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_2():
    my_cool_mesh = cm.RectangularMesh(nx = 4, ny = 8, nz = 2, units = 5e-6)
    return my_cool_mesh

@pytest.fixture(scope='module')
def mesh_2_cuboid():
    my_cool_mesh = cm.RectangularMesh(dx = 2, dy = 1, dz = 1, nx = 4, ny = 8, nz = 2, units = 5e-6)
    return my_cool_mesh


class TestMagnetisationField(object):
    """
    Test for the magnetisation field
    """
    def test_m_1(self, m_1):
        expected_flag, flag = True, False
        for i in range(m_1.shape[0]):
            for j in range(m_1.shape[1]):
                for k in range(m_1.shape[2]):
                    if np.isclose(np.linalg.norm(m_1[i, j, k]), 1):
                        flag = True
                    
        assert (m_1.shape == (8, 16, 4, 3))  # shape test
        assert (flag == expected_flag)  # magnetitude test
        
    
    
    def test_m_2(self, m_2):
        expected_flag, flag = True, False
        for i in range(m_2.shape[0]):
            for j in range(m_2.shape[1]):
                for k in range(m_2.shape[2]):
                    if np.isclose(np.linalg.norm(m_2[i, j, k]), 1):
                        flag = True
                        

        assert (m_2.shape == (4, 8, 2, 3))  # shape test
        assert (flag == expected_flag)  # magnetitude test




class TestEnergyTerm(object):
    """
    Test for Energy terms
    """
    @pytest.mark.parametrize("mesh, m, A, Ms", [
        ("mesh_1", "m_1", 0, 8e5), 
        ("mesh_1", "m_1", 8.78e-12, 8e5), 
        ("mesh_1_cuboid", "m_1", 8.78e-12, 8e5),
        ("mesh_2", "m_2", 0, 3.84e5),
        ("mesh_2", "m_2", 8.78e-12, 3.84e5),
        ("mesh_2_cuboid", "m_2", 8.78e-12, 3.84e5)])
    def test_exchange(self, mesh, m, A, Ms, request):
        """
        Test the exchange energy
        """
        # my program --- energy
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a class instance of m_Field
        # my_cool_m = cm.m_Field(m, my_cool_mesh, Ms)
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)

        exchange_energy = cm.Exchange(A=A).energy(my_cool_m)
        exchange_Heff = cm.Exchange(A=A).effective_field(my_cool_m)
        w_exchange = cm.Exchange(A=A).energy_density(my_cool_m)
        
    
        # ubermag --- m_uber_4 total energy
        
        mesh_size = my_cool_mesh.mesh_size
        region = df.Region(p1=(0, 0, 0), p2=mesh_size)
        cell_size = my_cool_mesh.cell_size
        mesh = df.Mesh(region=region, cell=cell_size)
        
        m_uber = df.Field(mesh, dim=3, value=m, norm=Ms)
        system = mm.System(name='test_eng_exchange')
        system.m = m_uber
        system.energy = mm.Exchange(A=A)

        uber_exchange_energy = oc.compute(system.energy.energy, system)
        uber_exchange_Heff = oc.compute(system.energy.effective_field, system)
        uber_w_exchange = oc.compute(system.energy.density, system)
        
        assert np.isclose(exchange_energy, uber_exchange_energy, rtol=0.01)
        assert np.allclose(exchange_Heff, uber_exchange_Heff.array)
        assert np.allclose(w_exchange, uber_w_exchange.array)
   
        
    @pytest.mark.parametrize('mesh, m, D, Ms', [
        ("mesh_1", "m_1", 0, 8e5), 
        ("mesh_1", "m_1", 1e-3, 8e5),
        ("mesh_1_cuboid", "m_1", 1e-3, 8e5),
        ("mesh_2", "m_2", 0, 3.84e5),
        ("mesh_2", "m_2", 1.58e-3, 3.84e5),
        ("mesh_2_cuboid", "m_2", 1.58e-3, 3.84e5)])
    def test_dmi(self, mesh, m, D, Ms, request):
        """
        Test the dmi energy 
        """
        # my program
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a instance of m_Field
        # my_cool_m = cm.m_Field(m, my_cool_mesh, Ms)
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        
        dmi_energy = cm.DMI(D=D).energy(my_cool_m)
        dmi_Heff = cm.DMI(D=D).effective_field(my_cool_m)
        w_dmi = cm.DMI(D=D).energy_density(my_cool_m)
        
        
        # ubermag
        mesh_size = my_cool_mesh.mesh_size
        region = df.Region(p1=(0, 0, 0), p2=mesh_size)
        cell_size = my_cool_mesh.cell_size
        mesh = df.Mesh(region=region, cell=cell_size)
    
        m_uber = df.Field(mesh, dim=3, value=m, norm=Ms)
        system = mm.System(name='test_eng_dmi')
        system.m = m_uber
        system.energy = mm.DMI(D=D, crystalclass='T')

        uber_dmi_energy = oc.compute(system.energy.energy, system)
        uber_dmi_Heff = oc.compute(system.energy.effective_field, system)
        uber_w_dmi = oc.compute(system.energy.density, system)
        
        assert np.isclose(dmi_energy, uber_dmi_energy, rtol=0.01)
        assert np.allclose(dmi_Heff, uber_dmi_Heff.array)
        assert np.allclose(w_dmi, uber_w_dmi.array)
        
 
    @pytest.mark.parametrize('mesh, m, H, Ms', [
        ("mesh_1", "m_1", (0, 0, 0), 8e5), 
        ("mesh_1", "m_1", (0, 0, 1e6), 8e5),
        ("mesh_1_cuboid", "m_1", (0, 0, 1e6), 8e5),
        ("mesh_2", "m_2", (0, 0, 0), 3.84e5),
        ("mesh_2", "m_2", (0, 0, 0.1/(4 * np.pi * 1e-7)), 3.84e5),
        ("mesh_2_cuboid", "m_2", (0, 0, 0.1/(4 * np.pi * 1e-7)), 3.84e5)])
    def test_zeeman(self, mesh, m, H, Ms, request):
        """
        Test the zeeman energy 
        """
        # my program
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a instance of m_Field
        # my_cool_m = cm.m_Field(m, my_cool_mesh, Ms)
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        
        zeeman_energy = cm.Zeeman(H=H).energy(my_cool_m)
        zeeman_Heff = cm.Zeeman(H=H).effective_field(my_cool_m)
        w_zeeman = cm.Zeeman(H=H).energy_density(my_cool_m)
        
        
        # ubermag
        mesh_size = my_cool_mesh.mesh_size
        region = df.Region(p1=(0, 0, 0), p2=mesh_size)
        cell_size = my_cool_mesh.cell_size
        mesh = df.Mesh(region=region, cell=cell_size)
        
        m_uber = df.Field(mesh, dim=3, value=m, norm=Ms)
        system = mm.System(name='test_eng_zeeman')
        system.m = m_uber
        system.energy = mm.Zeeman(H=H)

        uber_zeeman_energy = oc.compute(system.energy.energy, system)
        uber_zeeman_Heff = oc.compute(system.energy.effective_field, system)
        uber_w_zeeman = oc.compute(system.energy.density, system)
        
        assert np.isclose(zeeman_energy, uber_zeeman_energy, rtol=0.01)
        assert np.allclose(zeeman_Heff, uber_zeeman_Heff.array)
        assert np.allclose(w_zeeman, uber_w_zeeman.array)


    @pytest.mark.parametrize('mesh, m, u, K, Ms', [
        ("mesh_1", "m_1", (np.sqrt(2)/2, 1/2, 1/2), 0, 8e5), 
        ("mesh_1", "m_1", (1, 0, 0), 0.2e6, 8e5),
        ("mesh_1_cuboid", "m_1", (np.sqrt(3)/2, 0, 1/2), 0.2e6, 8e5),
        ("mesh_2", "m_2", (0, 1, 0), 0, 3.84e5),
        ("mesh_2", "m_2", (np.sqrt(2)/2, 1/2, 1/2), 1e5, 3.84e5),
        ("mesh_2_cuboid", "m_2", (np.sqrt(3)/2, 0, 1/2), 1e5, 3.84e5)])
    def test_UniaxialAnisotropy(self, mesh, m, u, K, Ms, request):
        """
        Test the zeeman energy 
        """
        # my program
        my_cool_mesh = request.getfixturevalue(mesh)
        m = request.getfixturevalue(m)
        # my_cool_m is a instance of m_Field
        # my_cool_m = cm.m_Field(m, my_cool_mesh, Ms)
        my_cool_m = cm.m_Field(my_cool_mesh, Ms, m)
        
        Anisotropy_energy = cm.UniaxialAnisotropy(K=K, u=u).energy(my_cool_m)
        Anisotropy_Heff = cm.UniaxialAnisotropy(K=K, u=u).effective_field(my_cool_m)
        w_Anisotropy = cm.UniaxialAnisotropy(K=K, u=u).energy_density(my_cool_m)
        
        
        # ubermag
        mesh_size = my_cool_mesh.mesh_size
        region = df.Region(p1=(0, 0, 0), p2=mesh_size)
        cell_size = my_cool_mesh.cell_size
        mesh = df.Mesh(region=region, cell=cell_size)
        
        m_uber = df.Field(mesh, dim=3, value=m, norm=Ms)
        system = mm.System(name='test_eng_anisotropy')
        system.m = m_uber
        system.energy = mm.UniaxialAnisotropy(K=K, u=u)

        uber_Anisotropy_energy = oc.compute(system.energy.energy, system)
        uber_Anisotropy_Heff = oc.compute(system.energy.effective_field, system)
        uber_w_Anisotropy = oc.compute(system.energy.density, system)
        
        assert np.isclose(Anisotropy_energy, uber_Anisotropy_energy, rtol=0.01)
        assert np.allclose(Anisotropy_Heff, uber_Anisotropy_Heff.array)
        assert np.allclose(w_Anisotropy, uber_w_Anisotropy.array)

"""Module dealing with energy terms."""


import numpy as np
from abc import ABC, abstractclassmethod, abstractmethod
from textwrap import dedent



class EnergyTerm(ABC):
    """ Abstract parent class for energy terms"""
    @abstractmethod
    def effective_field(self, m):
        """
        The effective field is the local field felt by the magnetization.
        
        Parameters
        ----------
        m : array_like
            normalised magnetisation field, norm = 1

        """
        pass
    
    def energy_density(self, m):
        """
        Energy density is the amount of energy stored a given system or 
        region of space per unit volume.
        
        Parameters
        ----------
        m : array_like
            normalised magnetisation field, norm = 1

        """
    
        w = -0.5 * m._mu0 * m.Ms * np.multiply(m.array, self.effective_field(m))
        w = np.sum(w, axis = 3, keepdims = True)
        return w

    def energy(self, m):
        """
        Energy meaning a specific form of enengy term that presents
        in the system.
        
        Parameters
        ----------
        m : array_like
            normalised magnetisation field, norm = 1

        """
        
        dV = m.v
        
        return np.sum(self.energy_density(m) * dV)


class Exchange(EnergyTerm):
    """ Abstract class for exchange term """
    
    def __init__(self, A):
        """
        Initialise the exchange energy' attributes
        
        Parameters
        ----------
        A : float,  optional
            exchange energy constant (J/m).

        """
        self.A = A
    
    def effective_field(self, m):
        laplace, _ , _ = m.laplace()
        res = (2 * self.A) / (m._mu0 * m.Ms) * laplace

        return res
    
    def mc_results(self, m, mode, spin_loc=None, old_laplace_component=None, old_m_padding=None):
        """
        Calculate both effective field and energy for the exchange term
        Parameters
        ----------
        m : array_like
            normalised magnetisation field, norm = 1
        
        mode : int
            mode = 0, meaning the first step of monte carlo iteration. mode = 1,
            the rest steps of monte carlo simulation
        
        spin_loc: ndarray
            the selected coordinates of the spin
            
        
        old_laplace_component: tuple
            contains the components of the laplace operator in the previous state: 
            laplace_operator_x, laplace_operator_y and laplace_operator_z.

        old_m_padding: ndarray
            the extension of 'm' in the previous state.

        """
        if mode:
            # Only renew line laplace for x, y, z each time
            new_laplace_component, new_m_padding = m.laplace_iter(spin_loc, old_laplace_component, old_m_padding)
            # pass
        else:
            _, new_laplace_component, new_m_padding = m.laplace()
        
        # Instead of calling effective_field, density, energy using functions in above,
        # I calculate them directly, so that I do not need to calculate laplace again
        laplace = new_laplace_component[0] + new_laplace_component[1] + new_laplace_component[2]
        exchange_Heff = (2 * self.A) / (m._mu0 * m.Ms) * laplace 
        w = -0.5 * m._mu0 * m.Ms * np.multiply(m.array, exchange_Heff)
        w = np.sum(w, axis = 3, keepdims = True)
        dV = m.v
        exchange_energy = np.sum(w * dV)


        return exchange_Heff, exchange_energy, new_laplace_component, new_m_padding

class DMI(EnergyTerm):
    """ Abstract class for DMI term """
    
    def __init__(self, D):
        """
        Initialise the DMI energy' attributes
        
        Parameters
        ----------
        A : float,  optional
            D is the DMI constant.
        """
        if isinstance(D, (int, float)):
            self.D = D

        else:
            raise TypeError("D must be a number")

    def effective_field(self, m):
    
        curl, _, _ = m.curl()
        res  = -2 * self.D / (m._mu0 * m.Ms) * curl
        return res
    
    def mc_results(self, m, mode, spin_loc=None, old_curl_component=None, old_m_padding=None):
        """
        Calculate both effective field and energy for the DMI term
        
        Parameters
        ----------
        m : array_like
            normalised magnetisation field, norm = 1
        
        mode : int
            mode = 0, meaning the first step of monte carlo iteration. mode = 1,
            the rest steps of monte carlo simulation
        
        spin_loc: ndarray
            the selected coordinates of the spin
            
        old_curl_component: tuple
            contains the components of the curl operator in the previous state: 
            curl in x direction , curl in y direction and curl in z direction.

        old_m_padding: ndarray
            the extension of 'm' in the previous state.

        """
       
        if mode:
            # Only renew line curl for x, y, z each time
            new_curl_component, new_m_padding = m.curl_iter(spin_loc, old_curl_component, old_m_padding)

        else:
            _, new_curl_component, new_m_padding = m.curl()
        # Instead of calling effective_field, density, etic in above,
        # I calculate them directly, so that I do not need to calculate laplace again
        curl_operator = np.zeros(m.array.shape)
        curl_operator[:, :, :, 0] = new_curl_component[3] - new_curl_component[5]
        curl_operator[:, :, :, 1] = new_curl_component[4] - new_curl_component[1]
        curl_operator[:, :, :, 2] = new_curl_component[0] - new_curl_component[2]
        dmi_Heff = -2 * self.D / (m._mu0 * m.Ms) * curl_operator 
        w = -0.5 * m._mu0 * m.Ms * np.multiply(m.array, dmi_Heff)
        w = np.sum(w, axis = 3, keepdims = True)
        dV = m.v
        dmi_energy = np.sum(w * dV)

        return dmi_Heff, dmi_energy, new_curl_component, new_m_padding


class Zeeman(EnergyTerm):
    """ abstract class for zeeman term """
    
    def __init__(self, H):
        """
        Initialise the zeeman energy' attributes
        Parameters
        ----------
        H : (3,) array_like, optional
           external magnetic field (A/m).
        """
          
        if not isinstance(H, (np.ndarray, list, tuple)):
            raise TypeError("u must be numpy.ndarray, list or list")
        
        if np.array(H).size != 3:
            raise ValueError('dimension mismatch')
        
        self.H = np.array(H)
        
    def effective_field(self, m):
        Heff_zeeman = np.tile(self.H, m.array.shape[0] * m.array.shape[1] * m.array.shape[2])
        Heff_zeeman = Heff_zeeman.reshape((m.array.shape[0], m.array.shape[1], m.array.shape[2], 3))
        return Heff_zeeman
    
    def energy_density(self, m):
        w_zeeman = -m._mu0 * m.Ms * np.multiply(m.array, self.effective_field(m))
        w_zeeman = np.sum(w_zeeman, axis = 3, keepdims = True)

        return w_zeeman


class UniaxialAnisotropy(EnergyTerm):
    """abstract class for UniaxialAnisotropy term"""
    
    def __init__(self, K, u):
        """
        Initialise the UniaxialAnisotropy energy' attributes
        Parameters
        ----------
        K: float, optional
            uniaxial anisotropy constant (J/m3)
            
        u: (3,) array_like, optional
            uniaxial anisotropy axis
        """
    
        if not isinstance(u, (np.ndarray, list, tuple)):
            raise TypeError("u must be numpy.ndarray, list or list")
        
        if np.array(u).size != 3:
            raise ValueError('dimension mismatch')

        if not np.isclose(np.linalg.norm(np.array(u)), 1):
            
            if not (u == np.zeros(3)).all():
                u = u / np.linalg.norm(u)
    
            else:
                raise ValueError("u cannot be a zero vector")

        self.K = K
        self.u = np.array(u)


    def effective_field(self, m):
        sector = 2 * self.K / (m._mu0 * m.Ms) * m.array @ self.u
        sector =  np.expand_dims(sector, axis = 3)  # reshape operator
        u_expanded = np.tile(self.u, m.array.shape[0] * m.array.shape[1] * m.array.shape[2])
        u_expanded = u_expanded.reshape(m.array.shape[0], m.array.shape[1], m.array.shape[2], 3)  # reshape operator
        
        Heff_anisotropy = np.multiply(sector, u_expanded)
        
        return Heff_anisotropy

    def energy_density(self, m):
        w_anisotropy = self.K * (1 - np.square(m.array @ self.u))
        w_anisotropy = np.expand_dims(w_anisotropy, axis=3)
        return w_anisotropy

    def mc_results(self, m):
        """
        Calculate both effective field and energy for the UniaxialAnisotropy term
        
        Parameters
        ----------
        m : array_like
            normalised magnetisation field, norm = 1

        """
        sector = 2 * self.K / (m._mu0 * m.Ms) * m.array @ self.u
        sector =  np.expand_dims(sector, axis = 3)
        u_expanded = np.tile(self.u, m.array.shape[0] * m.array.shape[1] * m.array.shape[2])
        u_expanded = u_expanded.reshape(m.array.shape[0], m.array.shape[1], m.array.shape[2], 3)  
        Heff_anisotropy = np.multiply(sector, u_expanded)
        
        w_anisotropy = self.K * (1 - np.square(m.array @ self.u))
        w_anisotropy = np.expand_dims(w_anisotropy, axis=3)
        dV = m.v
        anisotropy_energy = np.sum(w_anisotropy * dV)

        return Heff_anisotropy, anisotropy_energy
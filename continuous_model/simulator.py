import numpy as np
import os
import copy
import random


# pytest
from .energy import UniaxialAnisotropy
from .field import m_Field
from .energy import Zeeman
from .energy import Exchange
from .energy import DMI
from .mesh import RectangularMesh


class Simulator:

    def __init__(self, A, D, H, K, u, T):
        """
        Monte Carlo simulator 
    
        Parameters
        ----------
        A : float,  optional
            exchange energy constant (J/m).
            
        D : float, optional
            D is the DMI constant.

        H : (3,) array_like, optional
           external magnetic field (A/m).
            
        T : float
            Temperature in Kelvin (K).
            
        K: float, optional
            uniaxial anisotropy constant (J/m3)
            
        u: (3,) array_like, optional
            uniaxial anisotropy axis

        """
        self.A = A
        self.D = D

        self.H = H
        self.T = T
        
        self.K = K
        self.u = u

        self.monte_carlo_algorithm = monte_carlo

    def compute_minimum(self, m_field, atol, maxiter, step=None, save_path=None):
        """
        Calculate the magnetic equilibrium state and its energy at this state.
        
        Parameters
        ----------
        m_field : continuous_model.m_Field
        
        atol: float, optional.
            accept the value when the current state when the dot product between magnetic field m and 
            effective field within a given tolerance.
            
        maxiter: int, optional
            sets a limit on the number of times magnetisation field will be 
            iterated when the Monte carlo is run.
        
        step: int, optional.
            the procedure will monitor the periodic magnetisation field changes.
            
        save_path: str, optional.
            the periodic magnetisation field changes will be saved in that dir.

        Returns
        -------

        final_state: ndarray
            The final energy state after the Monte Carlo simulation
            
        minimum_energy : float
            The lowest energy after the Monte Carlo simulation
            

        """
        final_state, minimum_energy = self.monte_carlo_algorithm(m_field, A=self.A, D=self.D,H=self.H,
                                                                 K=self.K, u=self.u, T=self.T,
                                                                 atol=atol, maxiter=maxiter, debug=True,
                                                                 step=step, save_path=save_path)

        return final_state, minimum_energy

def monte_carlo(m_field, **kwargs):
    def update(exchange_results, dmi_results, H_eff_zeeman, anisotropy_results):
        H_eff = 0
        if exchange_results is not None:
            laplace_component, laplace_padding = exchange_results[2], exchange_results[3]
            H_eff += exchange_results[0]
        else:
            laplace_component, laplace_padding = None, None
        if dmi_results is not None:
            curl_component, curl_padding = dmi_results[2], dmi_results[3]
            H_eff += dmi_results[0]
        else:
            curl_component, curl_padding = None, None
        if H_eff_zeeman is not None:
            H_eff += H_eff_zeeman
        if anisotropy_results is not None:
            H_eff += anisotropy_results[0]
            
        return H_eff, laplace_component, laplace_padding, curl_component, curl_padding

    # step 0 - Get all parameters
    kB = 1.3806485279e-23
    H = kwargs.get('H', None)  # default value None
    A = kwargs.get('A', None)  # default value None
    D = kwargs.get('D', None)  # default value None
    T = kwargs.get('T', None)  # default value None
    K = kwargs.get('K', None)  # default value None
    u = kwargs.get('u', None)  # default value None
    
    atol = kwargs.get('atol', 0.001) # default value 0.001
    maxiter = kwargs.get('maxiter', 20000)  # default value 20000
    debug = kwargs.get('debug', False)  # default debug mode is false
    step = kwargs.get('step', None)
    save_path = kwargs.get('save_path', None)

    if T is None:
        raise ValueError('T must be given')
    A_ignore_flag = (A is None) or (A == 0) 
    D_ignore_flag = (D is None) or (D == 0)
    H_ignore_flag = (H is None) or (H == 0) or (isinstance(H, list) and not any(H))
    Ani_ignore_flag = (K is None and u is None) or (K == 0) or (isinstance(u, list) and not any(u))

    # ignore if is None or is 0 or is list with all 0s
    if all([A_ignore_flag, D_ignore_flag, H_ignore_flag, Ani_ignore_flag]):
        raise AssertionError('Everything is ignored')
    if debug:
        if A_ignore_flag:
            print('A is ignored')
        if D_ignore_flag:
            print('D is ignored')
        if H_ignore_flag:
            print('H is ignored')
        if Ani_ignore_flag:
            print('K, u are ignored')
            

    # Step 1 - Initialise the spin system
    E_0 = 0

    # Step 2 - Compute the energy E_0
    # Define classes early
    if A_ignore_flag:
        exchange = None
    else:
        exchange = Exchange(A=A)
    if D_ignore_flag:
        dmi = None
    else:
        dmi = DMI(D=D)
    if H_ignore_flag:
        zeeman = None
    else:
        zeeman = Zeeman(H=H)
        
    if Ani_ignore_flag:
        anisotropy = None
    else:
        anisotropy = UniaxialAnisotropy(K=K, u=u)

    # avoiding the calculation of H_eff and E_0 is to repeatedly
    H_eff = 0  # effective field
    
    E_0 = 0  # inital energy
    
    if exchange:
        exchange_results = exchange.mc_results(m_field, 0)

        H_eff += exchange_results[0]
        E_0 += exchange_results[1]
    else:
        exchange_results = None
    if dmi:
        dmi_results = dmi.mc_results(m_field, 0)

        H_eff += dmi_results[0]
        E_0 += dmi_results[1]
    else:
        dmi_results = None
    if zeeman:
        H_eff_zeeman = zeeman.effective_field(m_field)
        E_zeeman = zeeman.energy(m_field)

        H_eff += H_eff_zeeman
        E_0 += E_zeeman
    else:
        H_eff_zeeman = None
        E_zeeman = None
        
    if anisotropy:
        anisotropy_results =  anisotropy.mc_results(m_field)
        
        H_eff += anisotropy_results[0]
        E_0 += anisotropy_results[1]
    else:
        anisotropy_results = None
    
    # record the laplace and curl tensors each time, and only need to update
    # line laplace and line curl along x, y, z axis from the selected point
    if exchange_results:
        laplace_component, laplace_padding = exchange_results[2], exchange_results[3]
    else:
        laplace_component, laplace_padding = None, None
    if dmi_results:
        curl_component, curl_padding = dmi_results[2], dmi_results[3]
    else:
        curl_component, curl_padding = None, None
        
    j = 1

    for i in range(maxiter):
        max_error = np.amax(np.abs(np.cross(m_field.array, H_eff)))
        # if (cnt == 0 or cnt == 999999):
        #     print("max error right now: ", max_error)

        if max_error < atol:
            break

        else:

            # Step 3 - Randomly pick an atom
            # Create several random numbers as coordinates

            x = random.randint(0, m_field.nx - 1)
            y = random.randint(0, m_field.ny - 1)
            z = random.randint(0, m_field.nz - 1)

            # copy the selected spin
            old_spin = copy.deepcopy(m_field.array[x, y, z])

            # Selcect this random atom
            
            e = 1e-3 * np.random.uniform(-1, 1, 3)
            new_spin = old_spin + e
            m_field.array[x, y, z] = new_spin / np.linalg.norm(new_spin)

            #  Step 4 - Compute the energy E_1, w and r
            E_1 = 0

            if exchange:
                exchange_results = exchange.mc_results(m_field, 1, (x, y, z), laplace_component, laplace_padding)
                E_1 += exchange_results[1]
            if dmi:
                dmi_results = dmi.mc_results(m_field, 1, (x, y, z), curl_component, curl_padding)
                E_1 += dmi_results[1]
            if zeeman:
                E_zeeman = zeeman.energy(m_field)
                E_1 += E_zeeman
            if anisotropy:
                anisotropy_results = anisotropy.mc_results(m_field)
                E_1 += anisotropy_results[1]
            
            # calculate delta energy right only have zeeman
            delta_E = E_1 - E_0

            # dealing with T = 0, which is a special case
            if (T == 0):
                if delta_E <= 0:
                    E_0 = E_1  # accept E_1
                    H_eff, laplace_component, laplace_padding, curl_component, curl_padding = update(exchange_results, dmi_results, H_eff_zeeman, anisotropy_results)
                else:
                    m_field.array[x, y, z] = old_spin  # restore the old field

            else:
                #  Step 5(optional) - Comparing r and w
                if delta_E <= 0:
                    E_0 = E_1  # accept E_1
                    H_eff, laplace_component, laplace_padding, curl_component, curl_padding = update(exchange_results, dmi_results, H_eff_zeeman, anisotropy_results)
                else:
                    w = np.exp((-1 /(kB*T)) * delta_E)
                    r = np.random.uniform(0, 1)
                    if r <= w:
                        E_0 = E_1  # accept E_1
                        H_eff, laplace_component, laplace_padding, curl_component, curl_padding = update(exchange_results, dmi_results, H_eff_zeeman, anisotropy_results)
                    else:
                        m_field.array[x, y, z] = old_spin  # restore the old field
            
            
        if step:
            if i % step == 0 and step != maxiter:
                np.save(save_path + str(j) + ".npy", m_field.array)
                j += 1
        
    if step:
        np.save(save_path + "final.npy", m_field.array)

    return m_field.array, E_0
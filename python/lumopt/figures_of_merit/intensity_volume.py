# This FOM targets the integrated electric intensity, i.e. |Ex|^2+|Ey|^2+|Ez|^2 in a given volume V
#
# The corresponding object that acts as both monitor in forward simulation and source in the adjoint is the 'FieldRegion' object

import sys
import math
import numpy as np
import scipy as sp
import scipy.constants
import lumapi                      # import Lumapi to control FDTD simulation data 

from lumopt.utilities.wavelengths import Wavelengths

class IntensityVolume(object):

    ## Place FieldRegion objects inside the simulation where you'd like to optimize field intensity
    ## Modify the object to override global monitor settings to specify a frequency to target
    ## Pass the names of the FieldRegion objects to the IntensityVolume use for the optimization
    def __init__(self, field_region_name):   
        self.monitor_name = field_region_name
        
        name = str(self.monitor_name)
        if not name:
            raise UserWarning('Provided field_region_name is invalid.')
        
        self.target_fom = 0.0
        self.multi_freq_src = False     # Initially look at single operating frequency per simulation. Eventually, this flag should not be necessary!
        
    def initialize(self, sim):          # this is called in optimization class (lines: 680 - 690)      
        
        ## Check that the field_region acually exists and is unique
        self.check_monitor(sim) 

    def make_forward_sim(self, sim):
        sim.fdtd.setnamed(self.monitor_name, 'source mode', False)

    def make_adjoint_sim(self, sim):
        sim.fdtd.setnamed(self.monitor_name, 'source mode', True)

    def can_make_adjoint_sim(self, sim):
        return bool(int(sim.fdtd.haveresult(self.monitor_name)))

    def check_monitor(self, sim):
        
        haveFieldMonitor = (sim.fdtd.getnamednumber(self.monitor_name) >= 1) & (sim.fdtd.getnamed(self.monitor_name, 'type') == 'FieldRegion')
        if not haveFieldMonitor:       
            raise UserWarning(f"No field region with the name {self.monitor_name} found.")
        
        if sim.fdtd.getnamednumber(self.monitor_name) > 1:       
            raise UserWarning(f"Field region with the name {self.monitor_name} is not unique.")
            

    ## This should be called after the forward simulation but before the adjoint simulation
    def get_fom(self, sim):                                

        self.wavelengths = IntensityVolume.get_wavelengths(sim)                                     # wl = [1,.....]
        self.T_fwd_vs_wavelength = IntensityVolume.fr_monitor_fom(sim, self.monitor_name)           # FOM Vs wl = [1, . . . ]

        return self.T_fwd_vs_wavelength
    
    
    ## Figure of Merit (FOM) -  Electric field intensity at a particular point (position of point monitor)
    ## Point FOM computes the intensity at the desired location  

    @staticmethod
    def fr_monitor_fom(sim, field_region_name):
        
        sim.fdtd.eval(  'Ex = getresult("' + field_region_name + '", "Ex");\
                        Ey = getresult("' + field_region_name + '", "Ey");\
                        Ez = getresult("' + field_region_name + '", "Ez");\
                        E_intensity = sum(abs(Ex)^2 + abs(Ey)^2 + abs(Ez)^2);')
        
        E_intensity = sim.fdtd.getv('E_intensity')

        return E_intensity                           

    # Useful formulations: 
    # Forward field at x0 is E(x0)
    # E_adjoint_field ==> point fom driven backward E-field, calculation using Green's function, field at x is G(x,x0)*E(x0)
    # scaling factor = permittivity * del_V * E_adjoint_field (for normalized source, accounts E(x0) already)
    # scaling factor = permittivity * del_V * complex conjugate E(x0) * E_adjoint_field (for unit source)
    ## del_V (2D) = 2.5e-15 (check later?)

    def get_adjoint_field_scaling(self, sim):

        ### Scaling factor is considered while doing parallel computation only. Since the adjoint source is unit source. Requires scaling through conjugate complex fields.  
               
        ### Example: scaling_factor = -np.ones(1)*np.conj(self.field_complex_vs_wavelength)   # scaling_factor vs wl = [1,....] 
        ### Scaling factor is NOT considered while doing sequential computation. Since the adjoint source is conjugate complex fields (Ex, Ey, Ez)

        ### Normalization (by 'base amplitude') is considered in both parallel and sequential computation. 
        ### For sequential computation - no scaling of the adjoint field 
        numWavelengths = int(sim.fdtd.getglobalmonitor('frequency points'))
        scaling_factor = - np.ones(numWavelengths) # [note: conj(E_complex is accounted through mag and phase angle)]
        
        return scaling_factor                         


    ## Get the currently configured wavelengths/frequencies
    @staticmethod
    def get_wavelengths(sim):
        return Wavelengths(sim.fdtd.getglobalsource('wavelength start'),               
                           sim.fdtd.getglobalsource('wavelength stop'),                
                           sim.fdtd.getglobalmonitor('frequency points')).asarray()    
    

    def fom_gradient_wavelength_integral_on_cad(self, sim, grad_var_name, wl):

        assert np.allclose(wl, self.wavelengths)

        T_fwd_error = self.T_fwd_vs_wavelength        # returns scalar value = + fom 

        sim.fdtd.eval(('T_fwd_partial_derivs=real({0});').format(grad_var_name))    # copies gradient vector to T_fwd_partial_derivs as variable in FDTD 

        sim.fdtd.eval(('f_from_{0} = getresult("{0}", "E field");\n\
                        f_from_{0} = f_from_{0}.f;\n\
                        T_fwd_partial_derivs_freq_index = find(adjoint_fields.E.f, f_from_{0});\n\
                        T_fwd_partial_derivs = T_fwd_partial_derivs(:,:,T_fwd_partial_derivs_freq_index);\n\
                        clear(f_from_{0});').format(self.monitor_name))

        T_fwd_partial_derivs_on_cad = sim.fdtd.getv("T_fwd_partial_derivs")       # creates a new physical variable in FDTD (to check plot - visual)
        T_fwd_partial_derivs_on_cad*= -1.0 * np.sign(T_fwd_error)          # this value does not reflect in FDTD because it changes in python program 

        return T_fwd_partial_derivs_on_cad.flatten()


    def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):

        assert np.allclose(wl, self.wavelengths)

        T_fwd_partial_derivs_vs_wl = -1.0 * np.real(T_fwd_partial_derivs_vs_wl)
        
        return T_fwd_partial_derivs_vs_wl.flatten()
       
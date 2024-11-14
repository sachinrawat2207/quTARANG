"""
MIT License

Copyright (c) 2024 Sachin Singh Rawat, Shawan Kumar Jha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from quTARANG.config.config import ncp
from quTARANG.src import evolution
from quTARANG.src.univ import fns, grid, data_io
from quTARANG.src.univ import fourier_transform as fft
from quTARANG.src.lib import vector_field as U
from quTARANG.util import fns_util as util
from pathlib import Path 
import quTARANG.config.mpara as para
import shutil
import os 
import sys
import time as tr

class GPE():
    """Main class for the simulation. 
    """
    def __init__(self, wfcfn = None, potfn = None):
        self.wfc = []
        self.wfck = []
        self.pot = []
        
        #list of all type of energies
        self.te = []
        self.ke = []
        self.kec = []
        self.kei = []
        self.pe = []
        self.qe = []
        self.ie = []
        self.chmpot = []
        self.estime = []
        
        self.xrms = []
        self.yrms = []
        self.zrms = []
        self.rrms = []
        self.srmstime = []
            
        self.xrms_temp = 0
        self.yrms_temp = 0
        self.zrms_temp = 0
        self.rrms_temp = 0
        self.wfcfn = wfcfn
        self.potfn = potfn 
        
        self.U = U.Vector_field()
        data_io.show_params()
        
        if para.device == "gpu":
            para.t0_c = ncp.cuda.Event()
            para.ti_c = ncp.cuda.Event()
            para.tf_c = ncp.cuda.Event()
            para.t0_c.record()
   
        elif para.device == "cpu":
            para.t0_c = tr.perf_counter()
        
        self.set_arrays()
        self.set_initfile()
        # For timedependent potential
        data_io.set_initcond(self)
        
        evolution.set_scheme()
        
        if para.device == 'gpu':
            para.ti_c.record()
            para.ti_c.synchronize()
            elapsed_time = ncp.cuda.get_elapsed_time(para.t0_c, para.ti_c) * 1e-3
        elif para.device == "cpu":
            para.ti_c = tr.perf_counter()
            elapsed_time = para.ti_c - para.t0_c
        print("Elapsed time(s): ", elapsed_time)
        print("***** Initialization completed *****")
        print("")
        
    def set_arrays(self):
        """Setup numpy/cupy arrays for wfc, wfck and pot for the simulation.
        """
        if para.dimension == 1:
            self.wfc = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.wfck = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.pot = ncp.zeros(para.Nx, dtype = para.real_dtype)
            
        elif para.dimension == 2:
            self.wfc = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.wfck = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.pot = ncp.zeros((para.Nx, para.Ny), dtype = para.real_dtype)
        
        elif para.dimension == 3:
            self.wfc = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)   
            self.wfck = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)      
            self.pot = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.real_dtype)  
    
        
    def set_initfile(self):
        data_io.gen_path()
        path = Path(para.op_path)/'para.py'
        
        if para.overwrite == False:
            if (path.is_file()):
                sys.exit("Data for a run is already in the output path. \nEither move whole data or change the folder!")
        
        else:
            shutil.copy(Path(os.path.realpath(__file__)).parents[3]/'para.py', Path(para.op_path)/'para.py')
            shutil.copy(Path(os.path.realpath(__file__)).parents[3]/'main.py', Path(para.op_path)/'main.py')
            
        if not (path.is_file()):
            shutil.copy(Path(os.path.realpath(__file__)).parents[3]/'para.py', Path(para.op_path)/'para.py')
            shutil.copy(Path(os.path.realpath(__file__)).parents[3]/'main.py', Path(para.op_path)/'main.py')

    # computes the norm
    def norm(self): 
        return fns.integralr(ncp.abs(self.wfc)**2)
    
    # renormalize the code
    def renorm(self, Npar = 1):   #C
        self.wfc = Npar * self.wfc/self.norm()**0.5
        
    def evolve(self):   
        evolution.time_advance(self)
        
    def compute_xrms(self):  
        self.xrms_temp = (fns.integralr(ncp.abs(self.wfc)**2 * grid.x_mesh**2) - (fns.integralr(ncp.abs(self.wfc)**2 * grid.x_mesh))**2)**.5

    def compute_yrms(self):  
        self.yrms_temp = (fns.integralr(ncp.abs(self.wfc)**2 * grid.y_mesh**2) - (fns.integralr(ncp.abs(self.wfc)**2 * grid.y_mesh))**2)**.5   
    
    def compute_zrms(self):  
        self.zrms_temp = (fns.integralr(ncp.abs(self.wfc)**2 * grid.z_mesh**2) - (fns.integralr(ncp.abs(self.wfc)**2 * grid.z_mesh))**2)**.5   

    def compute_rrms(self):
        if para.dimension == 2:   
            # self.rrms_temp = (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2)) - (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2)**0.5))**2)**.5 
            self.rrms_temp = (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2)))**.5 
            
        elif para.dimension == 3:
            # self.rrms_temp = (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2 + grid.z_mesh**2)) - (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2 + grid.z_mesh**2)**0.5)))**.5
            self.rrms_temp = (fns.integralr(ncp.abs(self.wfc)**2 * (grid.x_mesh**2 + grid.y_mesh**2 + grid.z_mesh**2)))**.5
            
    
    def compute_te(self):   #C
        self.U.temp[:] = fft.forward_transform(self.wfc)
        deriv = grid.volume * ncp.sum(grid.ksqr * ncp.abs(self.U.temp)**2) 
        return fns.integralr((self.pot + 0.5 * para.g * ncp.abs(self.wfc)**2) * ncp.abs(self.wfc)**2) + deriv/2
        
    def compute_chmpot(self):
        self.U.temp[:] = fft.forward_transform(self.wfc)
        deriv = grid.volume * ncp.sum(grid.ksqr * ncp.abs(self.U.temp)**2)
        return fns.integralr(((self.pot + para.g * ncp.abs(self.wfc)**2) * ncp.abs(self.wfc)**2)) + deriv/2
    
    def compute_qe(self): 
        fns.gradient(ncp.abs(self.wfc), self)
        if para.dimension == 1:
            self.U.temp[:] = 0.5 * (self.U.Vx**2)
            
        if para.dimension == 2:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2) 

        elif para.dimension == 3:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        # return fns.integralk()
        return fns.integralr(self.U.temp.real)


    def compute_ie(self):   #C
        return 0.5 * para.g * fns.integralr(ncp.abs(self.wfc)**4)
        
    
    def compute_pe(self): #C
        return fns.integralr(self.pot * ncp.abs(self.wfc)**2) 
    
    def _velocity1d(self):
        fns.gradient1d(self.wfc.conj(), self)
        self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2 
        return
    
    def _velocity2d(self):
        fns.gradient2d(self.wfc.conj(), self)
        self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2 
        self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/ncp.abs(self.wfc)**2
        return
    
    def _velocity3d(self):
        fns.gradient3d(self.wfc.conj(), self)
        self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2 
        self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/ncp.abs(self.wfc)**2 
        self.U.Vz[:] = -(self.wfc * self.U.Vz).imag/ncp.abs(self.wfc)**2
        return
    
    def _velocity(self):   
        fns.gradient(self.wfc.conj(), self)
        if para.dimension == 1:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2 
            
        if para.dimension == 2:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2 
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/ncp.abs(self.wfc)**2 
    
        elif para.dimension == 3:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/ncp.abs(self.wfc)**2 
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/ncp.abs(self.wfc)**2 
            self.U.Vz[:] = -(self.wfc * self.U.Vz).imag/ncp.abs(self.wfc)**2 
        return 
    
    def compute_ke(self):  # doubt while integrating in fourier space
        if para.dimension == 1:
            self._velocity1d()
            self.U.temp[:] = 0.5 * ncp.abs(self.wfc)**2 * (self.U.Vx**2)
        
        elif para.dimension == 2:
            self._velocity2d() 
            self.U.temp[:] = 0.5 * ncp.abs(self.wfc)**2 * (self.U.Vx**2 + self.U.Vy**2)    
        
        elif para.dimension == 3:    
            self._velocity3d() 
            self.U.temp[:] = 0.5 * ncp.abs(self.wfc)**2 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        return (fns.integralr(self.U.temp)).real
    
    def omegak1d(self):
        self._velocity1d() 
        self.U.temp[:] = ncp.abs(self.wfc)
        self.U.omegai_kx[:] = fft.forward_transform(self.U.temp * self.U.Vx)
        grid.ksqr[0] == 1
        self.U.temp[:] = (grid.kx_mesh * self.U.omegai_kx)/grid.ksqr
        
        # Compressible part calculation
        self.U.Vx[:] = grid.kx_mesh * self.U.temp
        grid.ksqr[0, 0] = 0 
            
        #incompressible part calculation
        self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
        return
    
    def omegak2d(self):  #C
        self._velocity2d() 
        self.U.temp[:] = ncp.abs(self.wfc)
        self.U.omegai_kx[:] = fft.forward_transform(self.U.temp * self.U.Vx)
        self.U.omegai_ky[:] = fft.forward_transform(self.U.temp * self.U.Vy)
        
        grid.ksqr[0, 0] = 1
        self.U.temp[:] = (grid.kx_mesh * self.U.omegai_kx + grid.ky_mesh * self.U.omegai_ky)/grid.ksqr

        # Compressible part calculation
        self.U.Vx[:] = grid.kx_mesh * self.U.temp
        self.U.Vy[:] = grid.ky_mesh * self.U.temp       
        grid.ksqr[0, 0] = 0   
        
        #incompressible part calculation
        self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
        self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy
        return
    
    def omegak3d(self):
        self._velocity3d() 
        self.U.temp[:] = ncp.abs(self.wfc)
        self.U.omegai_kx[:] = fft.forward_transform(self.U.temp * self.U.Vx)
        self.U.omegai_ky[:] = fft.forward_transform(self.U.temp * self.U.Vy)
        self.U.omegai_kz[:] = fft.forward_transform(self.U.temp * self.U.Vz)
        
        grid.ksqr[0, 0, 0] = 1 
        self.U.temp[:] = (grid.kx_mesh * self.U.omegai_kx + grid.ky_mesh * self.U.omegai_ky + grid.kz_mesh * self.U.omegai_kz)/grid.ksqr
        
        # Compressible part calculation
        self.U.Vx[:] = grid.kx_mesh * self.U.temp
        self.U.Vy[:] = grid.ky_mesh * self.U.temp
        self.U.Vz[:] = grid.kz_mesh * self.U.temp
        grid.ksqr[0, 0, 0] = 0 
        
        #incompressible part calculation
        self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
        self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy
        self.U.omegai_kz[:] = self.U.omegai_kz - self.U.Vz
        return
    
    def omegak(self):   
        self._velocity() 
        self.U.temp[:] = ncp.abs(self.wfc)
        self.U.omegai_kx[:] = fft.forward_transform(self.U.temp * self.U.Vx)
        if para.dimension == 1:
            grid.ksqr[0] == 1
            self.U.temp[:] = (grid.kx_mesh * self.U.omegai_kx)/grid.ksqr
            
            # Compressible part calculation
            self.U.Vx[:] = grid.kx_mesh * self.U.temp
            grid.ksqr[0, 0] = 0 

            #incompressible part calculation
            self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
            
            
        elif para.dimension == 2:
            grid.ksqr[0, 0] = 1
            self.U.omegai_ky[:] = fft.forward_transform(self.U.temp * self.U.Vy)
            self.U.temp[:] = (grid.kx_mesh * self.U.omegai_kx + grid.ky_mesh * self.U.omegai_ky)/grid.ksqr

            # Compressible part calculation
            self.U.Vx[:] = grid.kx_mesh * self.U.temp
            self.U.Vy[:] = grid.ky_mesh * self.U.temp       
            grid.ksqr[0, 0] = 0   
            
            #incompressible part calculation
            self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
            self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy
        
        elif para.dimension == 3:
            x = 0
            self.U.omegai_ky[:] = fft.forward_transform(self.U.temp * self.U.Vy)
            self.U.omegai_kz[:] = fft.forward_transform(self.U.temp * self.U.Vz)
            grid.ksqr[0, 0, 0] = 1 
            self.U.temp[:] = (grid.kx_mesh * self.U.omegai_kx + grid.ky_mesh * self.U.omegai_ky + grid.kz_mesh * self.U.omegai_kz)/grid.ksqr
    
            # Compressible part calculation
            self.U.Vx[:] = grid.kx_mesh * self.U.temp
            self.U.Vy[:] = grid.ky_mesh * self.U.temp
            self.U.Vz[:] = grid.kz_mesh * self.U.temp
            grid.ksqr[0, 0, 0] = 0 
            
            #incompressible part calculation
            self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
            self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy
            self.U.omegai_kz[:] = self.U.omegai_kz - self.U.Vz
        return 

    
    def ke_dec(self):    #C
        if para.dimension == 1:
            self.omegak1d()
            kec = 0.5 * fns.integralk(ncp.abs(self.U.Vx)**2)
            kei = 0.5 * fns.integralk(ncp.abs(self.U.omegai_kx)**2)
            
        elif para.dimension == 2:
            self.omegak2d()
            kec = 0.5 * fns.integralk(ncp.abs(self.U.Vx)**2 + ncp.abs(self.U.Vy)**2)
            kei = 0.5 * fns.integralk(ncp.abs(self.U.omegai_kx)**2 + ncp.abs(self.U.omegai_ky)**2)
        
        elif para.dimension == 3:
            self.omegak3d()
            kec = 0.5 * fns.integralk(ncp.abs(self.U.Vx)**2 + ncp.abs(self.U.Vy)**2 + ncp.abs(self.U.Vz)**2)
            kei = 0.5 * fns.integralk(ncp.abs(self.U.omegai_kx)**2 + ncp.abs(self.U.omegai_ky)**2 + ncp.abs(self.U.omegai_kz)**2)
        return kec, kei 
        
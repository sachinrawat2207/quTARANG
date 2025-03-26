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

from pathlib import Path 
import sys
import re
import h5py as hp 
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path.cwd().parents[0]))


from src.config import *

linewidth = 1.5
plt.rcParams.update({'font.size':'20',
                     'font.family':'serif',
                     'font.weight':'bold',
                     'lines.linewidth':linewidth,
                     'text.usetex':useLatex})

if device == "gpu":
    import cupy as ncp 
    import cupy.fft as fft
    from cupyx.scipy.special import j0
    
else:
    import numpy as ncp
    import numpy.fft as fft
    from scipy.special import j0
    
class specanalysis():
    def __init__(self):
        self.set_grid()
        self.op = op_dir
        self.wfcpath = Path(path)/'wfc'
    
    def sortfile(self, Ti, Tf):
        filenames = np.array(os.listdir(self.wfcpath))       
        l=[]
        filearray = []
        for i in filenames:
            u = re.findall(r'\d+', i)
            t_initial = float(u[0]+'.'+u[1])
            l.append(t_initial)
            
        sortedarg = np.argsort(l)[:]
        filearray = filenames[sortedarg]
        l = np.array(l)
        l = l[sortedarg]
        
        if Tf != None: 
            return filearray[np.where((l<= Tf) & (l>=Ti))]
        
        else:
            return filearray[np.where(l==Ti)]
    
    def load_pot(self):
        f = hp.File(Path(path)/'pot.h5', 'r')
        if device == 'gpu':
            self.pot = ncp.asarray(f['pot'][:])
        
        elif device == "cpu":
            self.pot = f['pot'][:]
        f.close()
        
    def get_file(self, wfcname):
        f = hp.File(self.wfcpath/wfcname,'r')
        wfc = f['wfc'][:]
        f.close()
        if device == 'gpu':
            wfc = ncp.asarray(wfc) 
        return wfc  
        
    def get_time(self, file_name):
        u = re.findall(r'\d+', file_name)
        t = float(u[0]+'.'+u[1])
        return t
    
    def set_grid(self):
        self.dx = para.Lx/para.Nx
        self.dkx = 2 * ncp.pi/para.Lx

        self.dy = para.Ly/para.Ny
        self.dky = 2 * ncp.pi/para.Ly

        self.dz = para.Lz/para.Nz 
        self.dkz = 2 * ncp.pi/para.Lz

        if dimension == 2:
            # Grid formation
            self.x = ncp.arange(-para.Nx//2, para.Nx//2) 
            self.kx = 2 * ncp.pi * ncp.roll(self.x, para.Nx//2)/para.Lx
            self.x = self.x * self.dx
            
            self.y = ncp.arange(-para.Ny//2, para.Ny//2) 
            self.ky = 2 * ncp.pi * ncp.roll(self.y, para.Ny//2)/para.Ly
            self.y = self.y * self.dy
            self.kx_mesh, self.ky_mesh = ncp.meshgrid(self.kx, self.ky, indexing = 'ij')
            self.ksqr = self.kx_mesh**2 + self.ky_mesh**2 

        elif dimension == 3:
            # Grid formation
            self.x = ncp.arange(-para.Nx//2, para.Nx//2) 
            self.kx = 2 * ncp.pi * ncp.roll(self.x, para.Nx//2)/para.Lx
            self.x = self.x * self.dx
            
            self.y = ncp.arange(-para.Ny//2, para.Ny//2) 
            self.ky = 2 * ncp.pi * ncp.roll(self.y, para.Ny//2)/para.Ly
            self.y = self.y * self.dy
            
            self.z = ncp.arange(-para.Nz//2, para.Nz//2)
            self.kz = 2 * ncp.pi * ncp.roll(self.z, para.Nz//2)/para.Lz
            self.z = self.z * self.dz
            self.kx_mesh, self.ky_mesh, self.kz_mesh = ncp.meshgrid(self.kx, self.ky, self.kz, indexing = 'ij')
            self.ksqr = self.kx_mesh**2 + self.ky_mesh**2 + self.kz_mesh**2
            
    def fft(self, wfc):
        return fft.fftn(wfc)/(para.Nx * para.Ny * para.Nz)
    
    def ifft(self, wfck):
        return fft.ifftn(wfck) * para.Nx * para.Ny * para.Nz
    
    def gradient2D(self, quant):
        quantk = self.fft(quant)
        return self.ifft(1j*self.kx_mesh*quantk), self.ifft(1j*self.ky_mesh*quantk)
    
    def gradient3D(self, quant):
        quantk = self.fft(quant)
        return self.ifft(1j*self.kx_mesh*quantk), self.ifft(1j*self.ky_mesh*quantk), self.ifft(1j*self.kz_mesh*quantk)
    
    def velocity2D(self, PSI):
        PSIx, PSIy  = self.gradient2D(PSI)
        rho = ncp.abs(PSI)**2
        vx = ncp.imag(PSI.conjugate()*PSIx)/rho 
        vy = ncp.imag(PSI.conjugate()*PSIy)/rho 
        ncp.nan_to_num(vx, 0)
        ncp.nan_to_num(vy, 0)
        return vx, vy
    
    def velocity3D(self, PSI):
        PSIx, PSIy, PSIz  = self.gradient3D(PSI)
        rho = ncp.abs(PSI)**2
        vx = ncp.imag(PSI.conjugate()*PSIx)/rho
        vy = ncp.imag(PSI.conjugate()*PSIy)/rho
        vz = ncp.imag(PSI.conjugate()*PSIz)/rho
        ncp.nan_to_num(vx, 0)
        ncp.nan_to_num(vy, 0)
        ncp.nan_to_num(vz, 0)
        return vx, vy, vz
    
    def helmholtzk2D(self, wx, wy):
        wx_bar =self.fft(wx) 
        wy_bar =self.fft(wy)
        
        self.ksqr[0, 0] = 1
        ## Compressible
        wxc_bar = (self.kx_mesh*self.kx_mesh*wx_bar + self.kx_mesh*self.ky_mesh*wy_bar)/self.ksqr
        wyc_bar = (self.ky_mesh*self.kx_mesh*wx_bar + self.ky_mesh*self.ky_mesh*wy_bar)/self.ksqr

        self.ksqr[0, 0] = 0

        ## Incompressible
        wxi_bar = wx_bar - wxc_bar 
        wyi_bar = wy_bar - wyc_bar

        Wi, Wc = [wxi_bar, wyi_bar], [wxc_bar, wyc_bar]
        return Wi, Wc 
    
    def helmholtzk3D(self, wx, wy, wz):
        wx_bar =self.fft(wx)
        wy_bar =self.fft(wy)
        wz_bar =self.fft(wz)
        
        self.ksqr[0, 0, 0] = 1
        ## Compressible
        wxc_bar = (self.kx_mesh*self.kx_mesh*wx_bar + self.kx_mesh*self.ky_mesh*wy_bar + self.kx_mesh*self.kz_mesh*wz_bar)/self.ksqr
        wyc_bar = (self.ky_mesh*self.kx_mesh*wx_bar + self.ky_mesh*self.ky_mesh*wy_bar + self.ky_mesh*self.kz_mesh*wz_bar)/self.ksqr
        wzc_bar = (self.kz_mesh*self.kx_mesh*wx_bar + self.kz_mesh*self.ky_mesh*wy_bar + self.kz_mesh*self.kz_mesh*wz_bar)/self.ksqr
        
        self.ksqr[0, 0, 0] = 0
        
        ## Incompressible
        wxi_bar = wx_bar - wxc_bar
        wyi_bar = wy_bar - wyc_bar
        wzi_bar = wz_bar - wzc_bar
        
        Wi, Wc = [wxi_bar, wyi_bar, wzi_bar], [wxc_bar, wyc_bar, wzc_bar]
        return Wi, Wc
    
    def helmholtz2D(self, wx, wy):
        wx_bar =self.fft(wx) 
        wy_bar =self.fft(wy)
        
        self.ksqr[0, 0] = 1
        ## Compressible
        wxc_bar = (self.kx_mesh*self.kx_mesh*wx_bar + self.kx_mesh*self.ky_mesh*wy_bar)/self.ksqr
        wyc_bar = (self.ky_mesh*self.kx_mesh*wx_bar + self.ky_mesh*self.ky_mesh*wy_bar)/self.ksqr

        self.ksqr[0, 0] = 0
        wxc = self.ifft(wxc_bar)
        wyc = self.ifft(wyc_bar)

        ## Incompressible
        wxi_bar = wx_bar - wxc_bar 
        wyi_bar = wy_bar - wyc_bar 

        wxi = self.ifft(wxi_bar)
        wyi = self.ifft(wyi_bar)
        
        Wi, Wc = [wxi, wyi], [wxc, wyc]
        return Wi, Wc
    
    def helmholtz3D(self, wx, wy, wz):
        wx_bar =self.fft(wx)
        wy_bar =self.fft(wy)
        wz_bar =self.fft(wz)
        
        self.ksqr[0, 0, 0] = 1
        ## Compressible
        wxc_bar = (self.kx_mesh*self.kx_mesh*wx_bar + self.kx_mesh*self.ky_mesh*wy_bar + self.kx_mesh*self.kz_mesh*wz_bar)/self.ksqr
        wyc_bar = (self.ky_mesh*self.kx_mesh*wx_bar + self.ky_mesh*self.ky_mesh*wy_bar + self.ky_mesh*self.kz_mesh*wz_bar)/self.ksqr
        wzc_bar = (self.kz_mesh*self.kx_mesh*wx_bar + self.kz_mesh*self.ky_mesh*wy_bar + self.kz_mesh*self.kz_mesh*wz_bar)/self.ksqr
        
        self.ksqr[0, 0, 0] = 0
        
        wxc = self.ifft(wxc_bar)
        wyc = self.ifft(wyc_bar)
        wzc = self.ifft(wzc_bar)
        
        ## Incompressible
        wxi_bar = wx_bar - wxc_bar
        wyi_bar = wy_bar - wyc_bar
        wzi_bar = wz_bar - wzc_bar
        
        wxi, wyi, wzi = self.ifft(wxi_bar), self.ifft(wyi_bar), self.ifft(wzi_bar)
        
        Wi, Wc = [wxi, wyi, wzi], [wxc, wyc, wzc]
        return Wi, Wc
    
    def zeropad2D(self, arr):
        if arr.shape[0]%2 or arr.shape[1]%2:
            raise ValueError("Dimension of array should be even in all directions.")
        return ncp.pad(arr, (arr.shape[0]//2, arr.shape[1]//2))
    
    def zeropad3D(self, arr):
        if arr.shape[0]%2 or arr.shape[1]%2 or arr.shape[2]%2:
            raise ValueError("Dimension of array should be even in all directions.")
        return ncp.pad(arr, pad_width=((arr.shape[0]//2,arr.shape[0]//2), (arr.shape[1]//2,arr.shape[1]//2), (arr.shape[2]//2,arr.shape[2]//2)))
        
        
    def auto_correlate2D(self, q):
        phi = self.zeropad2D(q)
        chi = self.fft(phi)*(self.dx/ncp.sqrt(2*ncp.pi))**2
        res = self.ifft(ncp.abs(chi)**2)*(self.dkx*para.Nx)**2 * para.Nx * para.Ny
        return ncp.fft.fftshift(res) 
    
    def auto_correlate3D(self, q):
        phi = self.zeropad3D(q)
        chi = self.fft(phi)*(self.dx/ncp.sqrt(2*ncp.pi))**3
        res = self.ifft(ncp.abs(chi)**2)*(self.dkx*para.Nx)**3 * para.Nx * para.Ny * para.Nz
        return ncp.fft.fftshift(res)
    
    # def binning(self, qnt): 
    #     val = ncp.zeros(para.Nx//2-1)
    #     k = self.kx[:para.Nx//2-1]
    #     for i in range(para.Nx//2-1):
    #         z = ncp.where((self.ksqr**.5 >= self.kx[i]) & (self.ksqr**0.5 < self.kx[i+1]))        
    #         val[i] = ncp.sum(qnt[z].real)
    #     return k[1:], val[1:]
    def binning(self, qnt): 
        val = ncp.zeros(para.Nx//2-1)
        k = self.kx[:para.Nx//2]
        for i in range(para.Nx//2-1):
            z = ncp.where((self.ksqr**.5 >= self.kx[i]) & (self.ksqr**0.5 < self.kx[i+1]))        
            val[i] = ncp.sum(qnt[z].real)
        return k[1:], val[:]
    
    def _ke_spec_bin2D(self, wfc):
        vx, vy = self.velocity2D(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy
        
        Wi, Wc = self.helmholtzk2D(wx, wy)
        wix, wiy = Wi
        wcx, wcy = Wc
        
        keik = 0.5 * (ncp.abs(wix)**2 + ncp.abs(wiy)**2)
        keck = 0.5 * (ncp.abs(wcx)**2 + ncp.abs(wcy)**2)
        
        k, keik = self.binning(keik) 
        k, keck = self.binning(keck)
        return k, keik, keck
    
    def _ke_spec_bin3D(self, wfc):
        vx, vy, vz = self.velocity3D(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy
        wz = a * vz
        
        Wi, Wc = self.helmholtzk3D(wx, wy, wz)
        wix, wiy, wiz = Wi
        wcx, wcy, wcz = Wc
        
        keik = 0.5 * (ncp.abs(wix)**2 + ncp.abs(wiy)**2 + ncp.abs(wiz)**2)
        keck = 0.5 * (ncp.abs(wcx)**2 + ncp.abs(wcy)**2 + ncp.abs(wcz)**2)
        
        k, keik = self.binning(keik)
        k, keck = self.binning(keck)
        return k, keik, keck
        
    def _ke_spec_resolved2D(self, wfc, k):
        vx, vy = self.velocity2D(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy

        Wi, Wc = self.helmholtz2D(wx, wy)
        wcx, wcy = Wc
        wix, wiy = Wi
        
        Ccx = self.auto_correlate2D(wcx)
        Ccy = self.auto_correlate2D(wcy)
        Cix = self.auto_correlate2D(wix)
        Ciy = self.auto_correlate2D(wiy)
        
        Cc = 0.5 * (Ccx + Ccy)
        Ci = 0.5 * (Cix + Ciy)
        
        Nxmod, Nymod = 2*para.Nx, 2*para.Ny
        Lxmod = self.x[-1] - self.x[0] + self.dx
        Lymod = self.y[-1] - self.y[0] + self.dy
        xp = ncp.linspace(-Lxmod,Lxmod,Nxmod+1)[:Nxmod]
        yq = ncp.linspace(-Lymod,Lymod,Nymod+1)[:Nymod]

        xx_new, yy_new = ncp.meshgrid(xp, yq, indexing='xy')
        kernel = lambda k: j0(k * (xx_new**2 + yy_new**2)**0.5)
        
        Ekc = ncp.zeros_like(k)
        Eki = ncp.zeros_like(k)
        
        for i in range(len(Ekc)):
            Ekc[i] = ncp.sum((kernel(k[i])*Cc).real)
            Eki[i] = ncp.sum((kernel(k[i])*Ci).real)
            
        return  Eki*k*self.dx*self.dy/(2*ncp.pi), Ekc*k*self.dx*self.dy/(2*ncp.pi)
    
    def _ke_spec_resolved3D(self, wfc, k):
        vx, vy, vz = self.velocity3D(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy
        wz = a * vz
        
        Wi, Wc = self.helmholtz3D(wx, wy, wz)
        
        wix, wiy, wiz = Wi
        wcx, wcy, wcz = Wc
        
        Ccx = self.auto_correlate3D(wcx)
        Ccy = self.auto_correlate3D(wcy)
        Ccz = self.auto_correlate3D(wcz)
        
        Cix = self.auto_correlate3D(wix)
        Ciy = self.auto_correlate3D(wiy)
        Ciz = self.auto_correlate3D(wiz)
        Cc = 0.5 * (Ccx + Ccy + Ccz)
        Ci = 0.5 * (Cix + Ciy + Ciz)
        Nxmod, Nymod, Nzmod = 2*para.Nx, 2*para.Nx, 2*para.Nx
        
        Lxmod = self.x[-1] - self.x[0] + self.dx
        Lymod = self.y[-1] - self.y[0] + self.dy
        Lzmod = self.z[-1] - self.z[0] + self.dz
        
        xp = ncp.linspace(-Lxmod,Lxmod,Nxmod+1)[:Nxmod]
        yq = ncp.linspace(-Lymod,Lymod,Nymod+1)[:Nymod]
        zq = ncp.linspace(-Lzmod,Lzmod,Nzmod+1)[:Nzmod]
        
        xx_new, yy_new, zz_new = ncp.meshgrid(xp, yq, zq, indexing='ij')
        kernel = lambda k: ncp.sinc(k * ncp.sqrt(xx_new**2 + yy_new**2 + zz_new**2)/ncp.pi)
        Eki = ncp.zeros_like(k)
        Ekc = ncp.zeros_like(k)
        for i in range(len(Ekc)):
            Ekc[i] = ncp.sum((ncp.pi*kernel(k[i])*Cc).real)
            Eki[i] = ncp.sum((ncp.pi*kernel(k[i])*Ci).real)
        return  Eki*k**2*self.dx*self.dy*self.dz/(2*ncp.pi**2), Ekc**k**2*self.dx*self.dy*self.dz/(2*ncp.pi**2)

    def ke_spec(self, Ti = 0, Tf = None, type = 'bin', N = 1000):
        spec_kei = 0
        spec_kec = 0
        wfcnames = self.sortfile(Ti, Tf)
        length = len(wfcnames)
        for i in range(length):
            wfc = self.get_file(wfcnames[i])
            if type == 'bin':
                if dimension == 2:
                    k, kei, kec = self._ke_spec_bin2D(wfc)
                elif dimension == 3:
                    k, kei, kec = self._ke_spec_bin3D(wfc)

            if type == 'resolved':
                k = ncp.linspace(self.dkx, ncp.max(self.kx), N)
                if dimension == 2:
                    kei, kec = self._ke_spec_resolved2D(wfc, k)
                elif dimension == 3:
                    kei, kec = self._ke_spec_resolved3D(wfc, k)
                    
            spec_kei = spec_kei + kei
            spec_kec = spec_kec + kec
            
        if device == 'gpu':
            return k.get(), spec_kei.get()/(i + 1), spec_kec.get()/(i + 1)
        elif device == 'cpu':
            return k, spec_kei/(i + 1), spec_kec/(i + 1)
    
    def _par_spec(self, wfc):
        return self.binning(ncp.abs(self.fft(wfc))**2)
    
    def par_spec(self, Ti = 0, Tf = None, N = 1000):
        spec = 0
        wfcnames = self.sortfile(Ti, Tf)
        length = len(wfcnames)
        for i in range(length):
            wfc = self.get_file(wfcnames[i])
            k, parspec = self._par_spec(wfc)
            spec = spec + parspec
        if device == 'gpu':
            return k.get(), spec.get()/(i+1)
        elif device == 'cpu':
            return k, spec/(i+1)
    
    def evolve_TSSP(self, wfc):
        wfcn = wfc.copy()
        wfcn = wfcn * ncp.exp(-1j * 0.5 * para.dt * (self.pot + para.g * wfcn.conj() * wfcn))
        wfcn = ncp.fft.fftn(wfcn)
        wfcn = wfcn * ncp.exp(-1j * 0.5 * para.dt * self.ksqr)
        wfcn = ncp.fft.ifftn(wfcn)
        wfcn = wfcn * ncp.exp(-1j * 0.5 * para.dt * (self.pot + para.g * wfcn.conj() * wfcn))
        return wfcn

    def ke_flux(self, Ti = 0, Tf = None, type = 'resolved', N = 1000):
        flux_kei = 0
        flux_kec = 0
        self.load_pot()
        wfcnames = self.sortfile(Ti, Tf)
        length = len(wfcnames)
        for i in range(length):
            wfc = self.get_file(wfcnames[i])
            wfcn = self.evolve_TSSP(wfc)
            
            if type == 'bin':
                if dimension == 2:
                    k, keiini, kecini = self._ke_spec_bin2D(wfc)
                    k, keifin, kecfin = self._ke_spec_bin2D(wfcn)
                elif dimension == 3:
                    k, keiini, kecini = self._ke_spec_bin3D(wfc)
                    k, keifin, kecfin = self._ke_spec_bin2D(wfcn)
                
            if type == 'resolved':
                k = ncp.linspace(self.dkx, ncp.max(self.kx), N)
                if dimension == 2:
                    keiini, kecini = self._ke_spec_resolved2D(wfc, k)
                    keifin, kecfin = self._ke_spec_resolved2D(wfcn, k)
            
                elif dimension == 3:
                    keiini, kecini = self._ke_spec_resolved3D(wfc, k)
                    keifin, kecfin = self._ke_spec_resolved3D(wfcn, k)
                    
            flux_kei += ncp.cumsum(-(keifin-keiini)/para.dt)
            flux_kec += ncp.cumsum(-(kecfin-kecini)/para.dt) 

        if device == 'gpu':
            return k.get(), flux_kei.get()/(i+1), flux_kec.get()/(i+1)

        elif device == 'cpu':
            return k, flux_kei/(i+1), flux_kec/(i+1)
        
    def _tk(self, wfc):    
        temp = 2*(self.fft(para.g * wfc * ncp.abs(wfc)**2 + wfc * self.pot) * ncp.conjugate(self.fft(wfc))).imag
        return self.binning(temp)
    
    def par_flux(self, Ti = 0, Tf = None):
        self.load_pot()
        flux_Npar = 0
        wfcnames = self.sortfile(Ti, Tf)
        length = len(wfcnames)
        for i in range(length):
            wfc = self.get_file(wfcnames[i])
            k, tk = self._tk(wfc)
            flux_Npar += -ncp.cumsum(tk) * self.dkx
            
        if device == 'gpu':
            return k.get(), flux_Npar.get()/(i+1)

        elif device == 'cpu':
            return k, flux_Npar/(i+1)
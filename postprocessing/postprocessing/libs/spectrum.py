from pathlib import Path 
import sys
import numpy as np
import re
import h5py as hp 
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd().parents[0]))
from postprocessing.config import *

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
        f = hp.File(Path(path)/'pot.h5', 'r')
        if device == 'gpu':
            self.pot = ncp.asarray(f['pot'][:])
        
        elif device == "cpu":
            self.pot = f['pot'][:]

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

        if para.dimension == 1:
            self.x = ncp.arange(-para.Nx//2, para.Nx//2)
            self.kx = 2 * ncp.pi * ncp.roll(self.x, para.Nx//2)/para.Lx
            self.x = self.x * self.dx
            self.kx_mesh = self.kx
            self.ksqr = self.kx_mesh**2

        elif para.dimension == 2:
            # Grid formation
            self.x = ncp.arange(-para.Nx//2, para.Nx//2) 
            self.kx = 2 * ncp.pi * ncp.roll(self.x, para.Nx//2)/para.Lx
            self.x = self.x * self.dx
            
            self.y = ncp.arange(-para.Ny//2, para.Ny//2) 
            self.ky = 2 * ncp.pi * ncp.roll(self.y, para.Ny//2)/para.Ly
            self.y = self.y * self.dy
            self.kx_mesh, self.ky_mesh = ncp.meshgrid(self.kx, self.ky, indexing = 'ij')
            self.ksqr = self.kx_mesh**2 + self.ky_mesh**2 

        elif para.dimension == 3:
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
            
        #     volume = para.Lx * para.Ly * para.Lz
        #     dV = dx * dy * dz
        #     dkV = dkx * dky * dkz
        #     # Meshgrid formation
        #     x_mesh, y_mesh, z_mesh = ncp.meshgrid(x, y, z, indexing = 'ij')
            self.kx_mesh, self.ky_mesh, self.kz_mesh = ncp.meshgrid(self.kx, self.ky, self.kz, indexing = 'ij')
            self.ksqr = self.kx_mesh**2 + self.ky_mesh**2 + self.kz_mesh**2
            
    def get_file(self, wfcname):
        f = hp.File(self.wfcpath/wfcname,'r')
        wfc = f['wfc'][:]
        f.close()
        if device == 'gpu':
            wfc = ncp.asarray(wfc) 
        return wfc     

    def fft(self, wfc):
        return fft.fftn(wfc)/(para.Nx * para.Ny) 
    
    def ifft(self, wfck):
        return fft.ifftn(wfck) * para.Nx * para.Ny
    
    def binning(self, qnt): #C
        val = ncp.zeros(para.Nx//2)
        k = self.kx[0:para.Nx//2]
        for i in range(para.Nx//2):
            z = ncp.where((self.ksqr**.5 >= self.kx[i]) & (self.ksqr**0.5 < self.kx[i+1]))        
            val[i] = ncp.sum(qnt[z].real)
        return k, val
    
    def gradient(self, quant):
        quantk = self.fft(quant)
        return ncp.array([self.ifft(1j*self.kx_mesh*quantk), self.ifft(1j*self.ky_mesh*quantk)])
    
    
    def velocity(self, PSI):
        PSIx, PSIy  = self.gradient(PSI)
        rho = ncp.abs(PSI)**2
        vx = ncp.imag(PSI.conjugate()*PSIx)/rho 
        vy = ncp.imag(PSI.conjugate()*PSIy)/rho 
        ncp.nan_to_num(vx, 0)
        ncp.nan_to_num(vy, 0)
        return vx, vy
    
    
    def helmholtz(self, wx, wy):
        wx_bar =self.fft(wx) 
        wy_bar =self.fft(wy)
        
        self.ksqr[0, 0] = 1
        ## Compressible
        wxc_bar = (self.kx_mesh*self.kx_mesh*wx_bar + self.kx_mesh*self.ky_mesh*wy_bar)/self.ksqr
        wyc_bar = (self.ky_mesh*self.kx_mesh*wx_bar + self.ky_mesh*self.ky_mesh*wy_bar)/self.ksqr
        # wxc_bar[0,0] = 0
        # wyc_bar[0,0] = 0
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
    
    def helmholtzk(self, wx, wy):
        wx_bar =self.fft(wx) 
        wy_bar =self.fft(wy)
        
        self.ksqr[0, 0] = 1
        ## Compressible
        wxc_bar = (self.kx_mesh*self.kx_mesh*wx_bar + self.kx_mesh*self.ky_mesh*wy_bar)/self.ksqr
        wyc_bar = (self.ky_mesh*self.kx_mesh*wx_bar + self.ky_mesh*self.ky_mesh*wy_bar)/self.ksqr
        # wxc_bar[0,0] = 0
        # wyc_bar[0,0] = 0
        self.ksqr[0, 0] = 0

        ## Incompressible
        wxi_bar = wx_bar - wxc_bar 
        wyi_bar = wy_bar - wyc_bar 

        
        Wi, Wc = [wxi_bar, wyi_bar], [wxc_bar, wyc_bar]
        return Wi, Wc
    
    def _ke_spec(self, wfc):
        vx, vy = self.velocity(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy
        Wi, Wc = self.helmholtzk(wx, wy)
        wix, wiy = Wi
        wcx, wcy = Wc
        
        keik = 0.5 * (ncp.abs(wix)**2 + ncp.abs(wiy)**2)
        keck = 0.5 * (ncp.abs(wcx)**2 + ncp.abs(wcy)**2)
        
        k, keik = self.binning(keik) 
        k, keck = self.binning(keck)
        return k, keik, keck
    
    def _par_spec(self, wfc):
        return self.binning(ncp.abs(fft.fftn(wfc))**2)    
    
    def zeropad(self, arr):
        if arr.shape[0]%2 or arr.shape[1]%2:
            raise ValueError("Dimension of array should be even in all directions.")
        return ncp.pad(arr, (arr.shape[0]//2, arr.shape[1]//2))
    
    def auto_correlate(self, q):
        phi = self.zeropad(q)
        chi = self.fft(phi)*(self.dx/ncp.sqrt(2*ncp.pi))**2
        res = self.ifft(ncp.abs(chi)**2)*(self.dkx*para.Nx)**2
        return ncp.fft.fftshift(res)
    
    
    def _kei_wkspec(self, wfc, k):
        vx, vy = self.velocity(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy

        Wi, _ = self.helmholtz(wx, wy)
        wix, wiy = Wi

        Cx = self.auto_correlate(wix)
        Cy = self.auto_correlate(wiy)
        C = 0.5 * (Cx + Cy)
        
        Nxmod, Nymod = 2*para.Nx, 2*para.Ny
        Lxmod = self.x[-1] - self.x[0] + self.dx
        Lymod = self.y[-1] - self.y[0] + self.dy
        x_p = ncp.linspace(-Lxmod, Lxmod, Nxmod+1)[:Nxmod]
        y_q = ncp.linspace(-Lymod, Lymod, Nymod+1)[:Nymod]

        xx_new, yy_new = ncp.meshgrid(x_p, y_q, indexing='xy')
        kernel = lambda k: j0(k * (xx_new**2 + yy_new**2)**0.5)
        
        Eki = ncp.zeros_like(k)
        for i in range(len(Eki)):
            Eki[i] = ncp.sum((kernel(k[i])*C).real)
        
        return Eki*k*self.dx*self.dy/(2*ncp.pi)

    def _kec_wkspec(self, wfc, k):
        vx, vy = self.velocity(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy

        _, Wc = self.helmholtz(wx, wy)
        wcx, wcy = Wc

        Cx = self.auto_correlate(wcx)
        Cy = self.auto_correlate(wcy)
        C = 0.5 * (Cx + Cy)
        
        Nxmod, Nymod = 2*para.Nx, 2*para.Ny
        Lxmod = self.x[-1] - self.x[0] + self.dx
        Lymod = self.y[-1] - self.y[0] + self.dy
        x_p = ncp.linspace(-Lxmod,Lxmod,Nxmod+1)[:Nxmod]
        y_q = ncp.linspace(-Lymod,Lymod,Nymod+1)[:Nymod]

        xx_new, yy_new = ncp.meshgrid(x_p, y_q, indexing='xy')
        kernel = lambda k: j0(k * (xx_new**2 + yy_new**2)**0.5)
        
        Ekc = ncp.zeros_like(k)
        for i in range(len(Ekc)):
            Ekc[i] = ncp.sum((kernel(k[i])*C).real)
        
        return Ekc*k*self.dx*self.dy/(2*ncp.pi)

    def _ke_wkspec(self, wfc, k):
        vx, vy = self.velocity(wfc)
        a = ncp.abs(wfc)
        wx = a * vx
        wy = a * vy

        Wi, Wc = self.helmholtz(wx, wy)
        wcx, wcy = Wc
        wix, wiy = Wi
        
        Ccx = self.auto_correlate(wcx)
        Ccy = self.auto_correlate(wcy)
        Cix = self.auto_correlate(wix)
        Ciy = self.auto_correlate(wiy)
        
        Cc = 0.5 * (Ccx + Ccy)
        Ci = 0.5 * (Cix + Ciy)
        
        Nxmod, Nymod = 2*para.Nx, 2*para.Ny
        Lxmod = self.x[-1] - self.x[0] + self.dx
        Lymod = self.y[-1] - self.y[0] + self.dy
        x_p = ncp.linspace(-Lxmod,Lxmod,Nxmod+1)[:Nxmod]
        y_q = ncp.linspace(-Lymod,Lymod,Nymod+1)[:Nymod]

        xx_new, yy_new = ncp.meshgrid(x_p, y_q, indexing='xy')
        kernel = lambda k: j0(k * (xx_new**2 + yy_new**2)**0.5)
        
        Ekc = ncp.zeros_like(k)
        Eki = ncp.zeros_like(k)
        
        for i in range(len(Ekc)):
            Ekc[i] = ncp.sum((kernel(k[i])*Cc).real)
            Eki[i] = ncp.sum((kernel(k[i])*Ci).real)
            
        return  Eki*k*self.dx*self.dy/(2*ncp.pi), Ekc*k*self.dx*self.dy/(2*ncp.pi)
    
    
    def evolve_TSSP(self, wfc):
        wfcn = wfc.copy()
        wfcn = wfcn*ncp.exp(-1j*0.5*para.dt*(self.pot + para.g*wfcn.conj()*wfcn))
        wfcn = ncp.fft.fftn(wfcn)
        wfcn = wfcn*ncp.exp(-1j*0.5*para.dt*self.ksqr)
        wfcn = ncp.fft.ifftn(wfcn)
        wfcn = wfcn*ncp.exp(-1j*0.5*para.dt*(self.pot + para.g*wfcn.conj()*wfcn))
        return wfcn

    def ke_spec(self, Ti = 0, Tf = None, type = 'bin', N = 5000):
        spec_kei = 0
        spec_kec = 0
        wfcnames = self.sortfile(Ti, Tf)
        length = len(wfcnames)
        for i in tqdm(range(length)):
            wfc = self.get_file(wfcnames[i])
            if type == 'bin':
                k, kei, kec = self._ke_spec(wfc)
        
            if type == 'resolved':
                k = ncp.linspace(0, para.Nx * ncp.pi/para.Lx, N)
                print(k.shape)
                kei, kec = self._ke_wkspec(wfc, k)
                
            spec_kei = spec_kei + kei
            spec_kec = spec_kec + kec
                
        return k, spec_kei/(i+1), spec_kec/(i+1)
    
    def par_spec(self, Ti = 0, Tf = None, N = 5000):
        spec = 0
        wfcnames = self.sortfile(Ti, Tf)
        length = len(wfcnames)
        for i in range(length):
            wfc = self.get_file(wfcnames[i])
            k, parspec = self._par_spec(wfc)
            spec = spec + parspec
        return k, spec/(i+1)

    def ke_flux(self, Ti = 0, Tf = None, N = 5000):
        flux_kei = 0
        flux_kec = 0
        wfcnames = self.sortfile(Ti, Tf)
        k = ncp.linspace(0, para.Nx * ncp.pi/para.Lx, N)
        length = len(wfcnames)
        for i in tqdm(range(length)):
            wfc = self.get_file(wfcnames[i])
            wfcn = self.evolve_TSSP(wfc)
            k = ncp.linspace(0, para.Nx * ncp.pi/para.Lx, N)
            keiini, kecini = self._ke_wkspec(wfc, k)
            keifin, kecfin = self._ke_wkspec(wfcn, k)
            flux_kei += ncp.cumsum(-(keifin-keiini)/para.dt)
            flux_kec += ncp.cumsum(-(kecfin-kecini)/para.dt)       
        return k, flux_kei/(i+1), flux_kec/(i+1)
        
    def _tk(self, wfc):    
        temp = 2*(self.fft(para.g * wfc * ncp.abs(wfc)**2 + wfc * self.pot) * ncp.conjugate(self.fft(wfc))).imag
        return self.binning(temp)
    
    def par_flux(self, Ti = 0, Tf = None):
        flux_Npar = 0
        wfcnames = self.sortfile(Ti, Tf)
        length = len(wfcnames)
        for i in tqdm(range(length)):
            wfc = self.get_file(wfcnames[i])
            k, tk = self._tk(wfc)
            temp = ncp.zeros_like(tk)
            s = temp.shape[0]
            temp[0] = tk[0]
            for i in range(s):
                temp[i] = tk[i] + temp[i-1]
            flux_Npar += temp 
            delta_k = k[1] - k[0]
        return k, -delta_k*flux_Npar/(i+1)

    
    
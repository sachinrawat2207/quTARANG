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
import quTARANG.config.mpara as para
from quTARANG.src.univ import grid
from quTARANG.src import evolution

ncp.random.seed(0)


def test(G):
    gammay = 2
    eps1 = 2
    G.wfc[:] = ((gammay)**0.25/(ncp.pi * eps1)**(0.5)) * ncp.exp(-(gammay * grid.y_mesh**2 + grid.x_mesh**2)/(2*eps1)) + 0j
    G.pot[:] = (grid.x_mesh**2 + gammay**2 * grid.y_mesh**2)/2
    return 


def rp(G):  
    theta_0 = 1
    
    dk = 2*ncp.pi/para.Lx
    kx = ncp.fft.fftshift(ncp.arange(-para.Nx//2, para.Nx//2))*dk
    ky = ncp.arange(para.Ny//2+1)*dk

    kx_mesh, ky_mesh = ncp.meshgrid(kx, ky, indexing='ij')
    kmod = ncp.sqrt(kx_mesh**2 + ky_mesh**2)
    
    # Chooses theta_kx between -pi to pi
    theta_kx = 0.9999*ncp.pi * (ncp.random.random((para.Nx, para.Ny//2+1))*2 - 1)
    theta = theta_0 * ncp.exp(1j * theta_kx)
    theta[1:para.Nx//2,0] = ncp.conj(theta[para.Nx-1:para.Nx//2:-1, 0])

    z = ncp.where((kmod > 3*dk) | (kmod < dk))
    theta[z] = 0

    phase = ncp.fft.irfft2(theta)*para.Nx*para.Ny
    G.wfc[:] = ncp.exp(1j * phase)
    G.pot[:] = 0

    return 

############################################# Random Vortices ##################################################
def generate_random_vortices(number_of_vortices, min_vortex_sep, system_size, other_condition = lambda x: True):
    vorts = ncp.zeros(number_of_vortices, dtype=ncp.complex128)
    count = 0
    while count<number_of_vortices:
        candidate = (ncp.random.random() + 1j*ncp.random.random())*system_size- (system_size/2 + 1j*system_size/2)
        satisfied = True
        for i in range(count):
            dist = ncp.abs(vorts[i]-candidate)
            if dist<min_vortex_sep:
                satisfied = False
        if satisfied and other_condition(candidate):
            vorts[count] = candidate
            count += 1
    
    return vorts

def gen_phase_unifom(Nvorts, locs_pvorts, locs_nvorts, L):
    H = lambda s: ncp.where(s < 0, ncp.zeros(s.shape), ncp.ones(s.shape))

    xx, yy = grid.x_mesh, grid.y_mesh
    X, Y = (2*ncp.pi/L)*(xx + L/2), (2*ncp.pi/L)*(yy + L/2)
    phase = ncp.zeros_like(X)
    
    for i in range(Nvorts//2):
        # xplus, yplus = locs_pvorts[i]
        # xminus, yminus = locs_nvorts[i]
        xplus, yplus = locs_pvorts[i].real, locs_pvorts[i].imag
        xminus, yminus = locs_nvorts[i].real, locs_nvorts[i].imag

        xkminus, ykminus = (2*ncp.pi/L)*(xminus + L/2), (2*ncp.pi/L)*(yminus + L/2)
        xkplus, ykplus = (2*ncp.pi/L)*(xplus + L/2), (2*ncp.pi/L)*(yplus + L/2)
        Xkminus, Ykminus = X - xkminus, Y - ykminus
        Xkplus, Ykplus = X - xkplus, Y - ykplus

        temp_ = ncp.zeros_like(X)
        for n in range(-5, 6):
            temp_ += ncp.arctan(ncp.tanh((Ykminus + 2*ncp.pi*n)/2)*ncp.tan((Xkminus - ncp.pi)/2))\
                - ncp.arctan(ncp.tanh((Ykplus + 2*ncp.pi*n)/2)*ncp.tan((Xkplus - ncp.pi)/2))\
                + ncp.pi*(H(Xkplus) - H(Xkminus))
        temp_ -= (xkplus - xkminus)/(2*ncp.pi) * Y
        phase += temp_

    return ncp.angle(ncp.exp(1j*phase))



def random_vortices(G): 
    ##################################################################
    Nvorts = 20  # No of vortices
    ##################################################################
    min_vortex_sep = para.Lx/25
    vorts = generate_random_vortices(Nvorts, min_vortex_sep, para.Lx-min_vortex_sep)
    pvorts = vorts[:Nvorts//2]
    nvorts = vorts[Nvorts//2:]
    phase = gen_phase_unifom(Nvorts, pvorts, nvorts, para.Lx)
    G.wfc[:] = 1
    G.pot[:] = 0
    N = G.norm()**0.5
    print("Evolution started for phase imprinting")
    
    for i in (range(10000)):
        G.wfc[:] = ncp.abs(G.wfc)*ncp.exp(1j*phase) 
        evolution.time_adv_istrang(G)
        G.renorm(Npar = N)

############################################# Generate Vortex latice ##################################################
def vortex_lattice(G):
    ##################################################################
    Nvorts = 100 # No of vortices
    ##################################################################
    Nv = int(Nvorts**0.5)
    a1, a2 = -para.Lx//2, para.Ly//2
    xpos = ncp.linspace(a1, a2, Nv, endpoint=False)
    ypos = ncp.linspace(a1, a2, Nv, endpoint=False)
    Xpos, Ypos = ncp.meshgrid(xpos, ypos)
    vorts = (Xpos + 1j*Ypos).ravel()
    charges = ncp.hstack((ncp.ones((Nv * Nv//2, 1)), -ncp.ones((Nv * Nv//2, 1)))).ravel()
    # ## Alternate case
    charges_temp = charges.reshape((Nv, Nv))
    charges = (charges_temp * charges_temp.T).ravel()
    # vorts += 1*(ncp.random.random(vorts.shape) + 1j*ncp.random.random(vorts.shape) + 0.5 - 0.5j) ## add offset

    pvorts = vorts[charges>0]
    nvorts = vorts[charges<0]
    phase = gen_phase_unifom(Nvorts, pvorts, nvorts, para.Lx)
    
    G.wfc[:] = 1
    G.pot[:] = 0
    N = G.norm()**0.5
    print("Evolution started for phase imprinting")
    for i in (range(10000)):
        G.wfc[:] = ncp.abs(G.wfc)*ncp.exp(1j*phase) 
        evolution.time_adv_istrang(G)
        G.renorm(Npar = N)
    G.pot[:] = 0
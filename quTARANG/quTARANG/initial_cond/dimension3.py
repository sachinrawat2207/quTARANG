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

ncp.random.seed(1)

def rp(G):  
    # Generate the smoothed random phase 
    theta_0 = 1
    dk = 2*ncp.pi/para.Lx
    
    kx = ncp.fft.fftshift(ncp.arange(-para.Nx//2, para.Nx//2))*dk
    ky = ncp.fft.fftshift(ncp.arange(-para.Nx//2, para.Nx//2))*dk
    kz = ncp.arange(para.Ny//2+1)*dk

    kx_mesh, ky_mesh, kz_mesh = ncp.meshgrid(kx, ky, kz, indexing='ij')
    kmod = ncp.sqrt(kx_mesh**2 + ky_mesh**2+ kz_mesh**2)
    
    # Chooses theta_kx between -pi to pi
    theta_kx = 0.999 * ncp.pi * (ncp.random.random((para.Nx, para.Ny, para.Nz//2+1))*2 - 1)
    theta = theta_0 * ncp.exp(1j * theta_kx)
    
    theta[para.Nx-1:para.Nx//2:-1,para.Ny-1:para.Ny//2:-1,0] = ncp.conj(theta[1:para.Nx//2,1:para.Ny//2,0])
    theta[para.Nx-1:para.Nx//2:-1,para.Ny//2-1:0:-1,0] = ncp.conj(theta[1:para.Nx//2,para.Ny//2+1:para.Ny,0])

    theta[para.Nx-1:para.Nx//2:-1,0,0] = ncp.conj(theta[1:para.Nx//2,0,0])
    theta[0,para.Ny-1:para.Ny//2:-1,0] = ncp.conj(theta[0,1:para.Ny//2,0])


    z = ncp.where((kmod > 3*dk) | (kmod < dk))
    theta[z] = 0
    phase = ncp.fft.irfftn(theta)*para.Nx*para.Ny
    
    # normalize the phase between -4pi to 4pi
    arg_norm = 4*ncp.pi
    phase = 2*arg_norm*(phase - ncp.min(phase))/ncp.ptp(phase)-arg_norm
    
    G.wfc = ncp.exp(1j * phase)
    G.pot = 0
    
    return

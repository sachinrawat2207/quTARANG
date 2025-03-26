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
from cupyx.scipy.interpolate import RBFInterpolator
import numpy as np
from tqdm import tqdm

ncp.random.seed(1)

def rp(G):
    Ns = 4
    if para.Nx == 128:
        K = 8*ncp.pi
    elif para.Nx == 256:
        K = 16*ncp.pi
    else:
        K = 32*ncp.pi ## K = 32pi for N^3 = 512^3

    xs = ncp.linspace(0, para.Nx + 1, Ns + 1)
    ys = ncp.linspace(0, para.Ny + 1, Ns + 1)
    zs = ncp.linspace(0, para.Nz + 1, Ns + 1)
    xxs, yys, zzs = ncp.meshgrid(xs, ys, zs)
    phase_samples = ncp.random.uniform(-K, K, xxs.shape)

    ## Periodic Boundary Condition
    phase_samples[-1,:,:] = phase_samples[0,:,:]
    phase_samples[:,-1,:] = phase_samples[:,0,:]
    phase_samples[:,:,-1] = phase_samples[:,:,0]

    ## Creating data for RBFInterpolator to work.
    xobs = ncp.array((xxs, yys, zzs)).reshape((3,-1)).T
    yobs = phase_samples.reshape((1,-1)).T
    xflat = ncp.array(ncp.meshgrid(ncp.linspace(0, para.Nx-1, para.Nx), ncp.linspace(0, para.Ny-1, para.Ny), ncp.linspace(0, para.Nz-1, para.Nz))).reshape(3, -1)
    yflat = RBFInterpolator(xobs, yobs, kernel='cubic')(xflat.T)

    ## Removing last row
    phase_initial = yflat.reshape(para.Nx, para.Ny, para.Nz)
    G.wfc[:] = ncp.exp(1j*phase_initial)
    G.pot= -para.g

    return

def TG(G):
    ## ARGLE Parameters
    if para.Nx == 512:
        gd4 = 12
        xi = 0.008839
        dtau = 3.125e-3

    elif para.Nx == 256:
        gd4 = 6
        xi = 0.017678
        dtau = 6.25e-3
    else:
        gd4 = 3
        xi = 0.035355
        dtau = 1.25e-2

    
    tauf = 60
    

    vx =  ncp.sin(grid.x_mesh)*ncp.cos(grid.y_mesh)*ncp.cos(grid.z_mesh)
    vy = -ncp.cos(grid.x_mesh)*ncp.sin(grid.y_mesh)*ncp.cos(grid.z_mesh)
    vz =  ncp.zeros_like(vy)
    vsqr = (vx**2 + vy**2 + vz**2)

    lam = ncp.cos(grid.x_mesh) * ncp.sqrt(2*ncp.abs(ncp.cos(grid.z_mesh)))
    mu  = ncp.cos(grid.y_mesh) * ncp.sqrt(2*ncp.abs(ncp.cos(grid.z_mesh)))*ncp.sign(ncp.cos(grid.z_mesh))

    def psi_e(x, y, xi):
        k = ncp.sqrt(x**2 + y**2)
        return (x + 1j* y) * ncp.tanh(k/(np.sqrt(2) * xi))/k

    def psi_4(x, y, xi):
        return psi_e(x - 1/np.sqrt(2), y, xi) * psi_e(x, y - 1/np.sqrt(2), xi) * psi_e(x + 1/np.sqrt(2), y, xi) * psi_e(x, y + 1/np.sqrt(2), xi)

    G.wfc[:] = psi_4(lam, mu, xi)**gd4
    tau = np.arange(0, tauf, dtau)
    print("ARGLE has been started")
    for i_ in tqdm(range(len(tau))):
        evolution.ARGLE(G, dtau, vx, vy, vz, vsqr)

    G.pot[:] = 0

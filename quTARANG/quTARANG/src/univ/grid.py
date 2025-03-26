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
from quTARANG.util import fns_util as util
from pathlib import Path
import re
import sys 
import quTARANG.config.mpara as para

dx = para.Lx/para.Nx
dkx = 2 * ncp.pi/para.Lx

dy = para.Ly/para.Ny
dky = 2 * ncp.pi/para.Ly

dz = para.Lz/para.Nz 
dkz = 2 * ncp.pi/para.Lz

if para.dimension == 1:
    x = ncp.arange(-para.Nx//2, para.Nx//2) 
    kx = 2 * ncp.pi * ncp.roll(x, para.Nx//2)/para.Lx
    x = x * dx
    
    volume = para.Lx
    dV = dx
    sKv = dkx
    
    # Meshgrid formation
    x_mesh = x
    kx_mesh = kx
    ksqr = kx_mesh**2

elif para.dimension == 2:
    # Grid formation
    x = ncp.arange(-para.Nx//2, para.Nx//2) 
    kx = 2 * ncp.pi * ncp.roll(x, para.Nx//2)/para.Lx
    x = x * dx
    
    y = ncp.arange(-para.Ny//2, para.Ny//2) 
    ky = 2 * ncp.pi * ncp.roll(y, para.Ny//2)/para.Ly
    y = y * dy

    volume = para.Lx * para.Ly
    dV = dx * dy
    dkV = dkx * dky
    # Meshgrid formation
    x_mesh, y_mesh = ncp.meshgrid(x, y, indexing = 'ij')
    kx_mesh, ky_mesh = ncp.meshgrid(kx, ky, indexing = 'ij')
    ksqr = kx_mesh**2 + ky_mesh**2 

elif para.dimension == 3:
    # Grid formation
    x = ncp.arange(-para.Nx//2, para.Nx//2) 
    kx = 2 * ncp.pi * ncp.roll(x, para.Nx//2)/para.Lx
    x = x * dx
    
    y = ncp.arange(-para.Ny//2, para.Ny//2) 
    ky = 2 * ncp.pi * ncp.roll(y, para.Ny//2)/para.Ly
    y = y * dy
    
    z = ncp.arange(-para.Nz//2, para.Nz//2)
    kz = 2 * ncp.pi * ncp.roll(z, para.Nz//2)/para.Lz
    z = z * dz
    
    volume = para.Lx * para.Ly * para.Lz
    dV = dx * dy * dz
    dkV = dkx * dky * dkz
    # Meshgrid formation
    x_mesh, y_mesh, z_mesh = ncp.meshgrid(x, y, z, indexing = 'ij')
    kx_mesh, ky_mesh, kz_mesh = ncp.meshgrid(kx, ky, kz, indexing = 'ij')
    ksqr = kx_mesh**2 + ky_mesh**2 + kz_mesh**2
    
# No of iteration fo time step
t_initial = 0
nstep = int((para.tmax-t_initial)/para.dt)

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

from quTARANG.src.univ import grid
from quTARANG.config.config import ncp
from quTARANG.src.univ import fourier_transform as fft
import quTARANG.config.mpara as para

# Integral in real space
def xderivative(arr):
    return (ncp.roll(arr, -1, axis = 0) - ncp.roll(arr, 1, axis = 0))/(2 * grid.dx)
    
def yderivative(arr):
    return (ncp.roll(arr, -1, axis = 1) - ncp.roll(arr, 1, axis = 1))/(2 * grid.dy)

def zderivative(arr):
    return (ncp.roll(arr, -1, axis = 2) - ncp.roll(arr, 1, axis = 2))/(2 * grid.dz)

def laplacian(arr):
    pass

def integral_simpson(arr):
    pass


def integralr(arr):
    """Integrates an array in real space (Trapezoid method).


    Parameters
    ----------
    arr : Array
    grid : Grid

    Returns
    -------
    complex
    """
    return  grid.dV * ncp.sum(arr)

# Integeral in fourier space
def integralk(arrk):
    return grid.volume * ncp.sum(arrk)

#Calculates the gradient of the wavefunction    
def gradient1d(arr, G):
    G.U.temp[:] = fft.forward_transform(arr)
    G.U.Vx[:] = fft.inverse_transform(1j * grid.kx_mesh * G.U.temp) 
    return

def gradient2d(arr, G):
    G.U.temp[:] = fft.forward_transform(arr)
    G.U.Vx[:] = fft.inverse_transform(1j * grid.kx_mesh * G.U.temp)
    G.U.Vy[:] = fft.inverse_transform(1j * grid.ky_mesh * G.U.temp)
    return

def gradient3d(arr, G):
    G.U.temp[:] = fft.forward_transform(arr)
    G.U.Vx[:] = fft.inverse_transform(1j * grid.kx_mesh * G.U.temp)
    G.U.Vy[:] = fft.inverse_transform(1j * grid.ky_mesh * G.U.temp)
    G.U.Vz[:] = fft.inverse_transform(1j * grid.kz_mesh * G.U.temp)
    return

def gradient(arr, G):
    """Calculates the gradient using spectral method along different axes depending on dimensionality.
        The result is stored in the Vx, Vy, Vz arrays inside U.
        U is an attrubite of the GPE object.

    Parameters
    ----------
    arr : Array
    G : GPE object
    
    """
    if para.dimension == 1:
        G.U.temp[:] = fft.forward_transform(arr)
        G.U.Vx[:] = fft.inverse_transform(1j * grid.kx_mesh * G.U.temp) 
        
    elif para.dimension == 2:
        G.U.temp[:] = fft.forward_transform(arr)
        G.U.Vx[:] = fft.inverse_transform(1j * grid.kx_mesh * G.U.temp)
        G.U.Vy[:] = fft.inverse_transform(1j * grid.ky_mesh * G.U.temp)
    
    elif para.dimension == 3:
        G.U.temp[:] = fft.forward_transform(arr)
        G.U.Vx[:] = fft.inverse_transform(1j * grid.kx_mesh * G.U.temp)
        G.U.Vy[:] = fft.inverse_transform(1j * grid.ky_mesh * G.U.temp)
        G.U.Vz[:] = fft.inverse_transform(1j * grid.kz_mesh * G.U.temp)
    
    return

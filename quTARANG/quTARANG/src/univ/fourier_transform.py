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

from quTARANG.config.config import fft
import quTARANG.config.mpara as para

def forward_transform1D(psi, axis = -1):
    return fft.fft(psi, axis = axis)/(para.Nx)

def inverse_transform1D(psik, axis = -1):
    return fft.ifft(psik, axis = axis) * (para.Nx)

def forward_transform2D(psi):
    return fft.fftn(psi)/(para.Nx * para.Ny)

def inverse_transform2D(psik):
    return fft.ifftn(psik) * (para.Nx * para.Ny)

def forward_transform3D(psi):
    return fft.fftn(psi)/(para.Nx * para.Ny * para.Nz)

def inverse_transform3D(psik):
    return fft.ifftn(psik) * (para.Nx * para.Ny * para.Nz)

if para.dimension == 1:
    forward_transform = forward_transform1D
    inverse_transform = inverse_transform1D

elif para.dimension == 2:
    forward_transform = forward_transform2D
    inverse_transform = inverse_transform2D

elif para.dimension == 3:
    forward_transform = forward_transform3D
    inverse_transform = inverse_transform3D
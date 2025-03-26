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

from quTARANG.config.config import  ncp
import quTARANG.config.mpara as para

class Vector_field:
    """ It contains the variables which helps to calculate the different physical quantities of GPE class 
    """
    Vx = []
    Vy = []
    Vz = []
    # Used to calculate omegai
    omegai_kx = []
    omegai_ky = []
    omegai_ky = []
    
    # Temporary variable
    temp = []
    
    def __init__(self) -> None:
        self.set_arrays()
        
    def set_arrays(self) -> None:
        if para.dimension == 1:
            self.Vx = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.omegai_kx = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.temp = ncp.zeros(para.Nx, dtype = para.complex_dtype) # used
            self.temp1k = ncp.zeros(para.Nx, dtype = para.complex_dtype)
            self.temp2k = ncp.zeros(para.Nx, dtype = para.complex_dtype)

        elif para.dimension == 2:
            self.Vx = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.Vy = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.omegai_kx = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.omegai_ky = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp1k = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            self.temp2k = ncp.zeros((para.Nx, para.Ny), dtype = para.complex_dtype)
            
        elif para.dimension == 3:
            self.Vx = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.Vy = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.Vz = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.omegai_kx = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.omegai_ky = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.omegai_kz = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp1k = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)
            self.temp2k = ncp.zeros((para.Nx, para.Ny, para.Nz), dtype = para.complex_dtype)

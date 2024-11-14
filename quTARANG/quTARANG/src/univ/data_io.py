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
from quTARANG.src.univ import fourier_transform as fft
from quTARANG.src.univ import grid
from quTARANG.util import fns_util as util
from pathlib import Path
import h5py as hp
import os
import quTARANG.config.mpara as para
import time as tr

if para.dimension == 2 and para.inp_type == "pdf":
    from quTARANG.initial_cond import dimension2 as dim

elif para.dimension == 3 and para.inp_type == "pdf":
    from quTARANG.initial_cond import dimension3 as dim
    
def print_params(G, time=0, iter=0, error = 0, energy = 0):
    # print("Chemical Potential: ", G.compute_chmpot())
    if para.imgtime == True:
        print("Iteration: ", iter)
        print("Stoping criteria for energy: delta E<", para.delta)
        print('delta E: ', error)
    
    elif para.imgtime == False:
        print('Time step: ', round(time, 7))
    print('Chemical Potential: ', G.compute_chmpot())
    print('Energy: ', G.compute_te())
    G.compute_rrms()
    print('rms: ', G.rrms_temp)
    print('Particle Number: ', G.norm())   
    if para.device == 'gpu':
        para.tf_c.record()
        para.tf_c.synchronize()
        elapsed_time = ncp.cuda.get_elapsed_time(para.t0_c, para.tf_c) * 1e-3 
    elif para.device == "cpu":
        para.tf_c = tr.perf_counter()
        elapsed_time = para.tf_c - para.t0_c
    print("Elapsed time(s): ", elapsed_time)         
    print("-----------------------------------------\n")
    
def show_params():
    print('[Nx, Ny, Nz]: ', para.Nx, para.Ny, para.Nz)
    print('[Lx, Ly, Lz]:', para.Lx, para.Ly, para.Lz)
    print('dimension: ', para.dimension)
    print("g: ", para.g)
    print('Scheme:', para.scheme)
    print('Imaginary time:', para.imgtime)
    print('dt: ', para.dt)
    print('device: ', para.device)
    if para.inp_type == "pdf":
        print("inital condition type:", para.typ)
    if para.imgtime == True:
        print("Stoping criteria for energy: delta E<", para.delta)
    
    elif para.imgtime == False:
        print('Initial time: ', grid.t_initial)
        print('Final time: ', para.tmax)
        print('Total iterations:', grid.nstep)

# directory generation
def gen_path():
    if not Path(para.op_path).exists():
        os.makedirs(para.op_path)
    if not (Path(para.op_path)/'wfc').exists():   
        os.mkdir(Path(para.op_path)/'wfc')

def set_initcond(G):      
    if para.inp_type == "fun":
        G.wfc[:] = G.wfcfn()
        G.pot[:] = G.potfn(0)
    
    elif para.inp_type == "dat":
        f1 = hp.File(Path(para.in_path)/'wfc.h5', 'r')
        f2 = hp.File(Path(para.in_path)/'pot.h5', 'r')
        
        if para.device == 'gpu':
            G.wfc[:] = ncp.asarray(f1['wfc'])
            G.pot[:] = ncp.asarray(f2['pot'])
        else:
            G.wfc[:] = f1['wfc']
            G.wfc[:] = f2['pot']
        f1.close()
        f2.close()
    elif para.inp_type == "pdf":
        if para.typ == "rp":
            dim.rp(G)
        
        if para.dimension == 2:
            if para.typ == "vl":
                dim.vortex_lattice(G)
                
            elif para.typ == "rv":
                dim.random_vortices(G)
    if para.save_wfc:
        save_wfc(G,0)
        save_pot(G)

def save_pot(G):
    f1 = hp.File( Path(para.op_path)/('pot.h5'), 'w')
    if para.device == 'gpu':
        f1.create_dataset('pot', data = ncp.asnumpy(G.pot))
    elif para.device == 'cpu':
        f1.create_dataset('pot', data = G.pot)
    f1.close()
            
def save_wfc(G, t):
    if para.imgtime == True:
        f1 = hp.File( Path(para.op_path)/('wfc/' + 'wfc_itr%d.h5'%t), 'w')
        if para.device == 'gpu':
            f1.create_dataset('wfc', data = ncp.asnumpy(G.wfc))
        elif para.device == 'cpu':
            f1.create_dataset('wfc', data = G.wfc)
        f1.close()
    elif para.imgtime == False:
        f1 = hp.File( Path(para.op_path)/('wfc/' + 'wfc_t%1.6f.h5'%t), 'w')
        if para.device == 'gpu':
            f1.create_dataset('wfc', data = ncp.asnumpy(G.wfc))
        elif para.device == 'cpu':
            f1.create_dataset('wfc', data = G.wfc)
        f1.close()

def compute_rms(G, t):
    if para.scheme == 'RK4':
        G.wfc = fft.inverse_transform(G.wfck)
        
    if para.device == 'gpu':            
        if para.dimension >= 1:
            G.compute_xrms()
            G.xrms.append(ncp.asnumpy(G.xrms_temp))
            G.srmstime.append(ncp.asnumpy(t))
        
        if para.dimension >= 2:
            G.compute_yrms()
            G.compute_rrms()    
            G.yrms.append(ncp.asnumpy(G.yrms_temp))
            G.rrms.append(ncp.asnumpy(G.rrms_temp))
        
        if para.dimension == 3:
            G.compute_zrms()
            G.zrms.append(ncp.asnumpy(G.zrms_temp))
        
    elif para.device == 'cpu':
        if para.dimension >= 1:
            G.compute_xrms()
            G.xrms.append(G.xrms_temp)
            G.srmstime.append(t)

        if para.dimension >= 2:
            G.compute_yrms()
            G.compute_rrms()    
            G.yrms.append(G.yrms_temp)
            G.rrms.append(G.rrms_temp)
        
        if para.dimension == 3:
            G.compute_zrms()
            G.zrms.append(G.zrms_temp)

        
def save_rms(G):
    filename = util.new_filename(Path(para.op_path)/'rms.h5')
    f = hp.f = hp.File(filename, 'w')
    if para.dimension >= 1:
        f.create_dataset('xrms', data = G.xrms)
        f.create_dataset('t', data = G.srmstime)
    
    if para.dimension >= 2:
        f.create_dataset('yrms', data = G.yrms)
        f.create_dataset('rrms', data = G.rrms)
        
    if para.dimension == 3:
        f.create_dataset('zrms', data = G.zrms)
    f.close()


def compute_energy(G, t):
    if para.device == 'gpu':
        G.te.append(ncp.asnumpy(G.compute_te()))
        G.chmpot.append(ncp.asnumpy(G.compute_chmpot()))
        ckec, ckei = G.ke_dec()
        G.ke.append(ncp.asnumpy(ckei + ckec))
        G.kei.append(ncp.asnumpy(ckei))
        G.kec.append(ncp.asnumpy(ckec))
        G.qe.append(ncp.asnumpy(G.compute_qe()))
        G.ie.append(ncp.asnumpy(G.compute_ie()))
        G.pe.append(ncp.asnumpy(G.compute_pe()))
            
    elif para.device == 'cpu':
        G.te.append(G.compute_chmpot())
        G.chmpot.append(G.compute_te())
        # ckec, ckei = G.ke_dec()
        G.ke.append(ckec + ckei)
        G.kei.append(ckei)
        G.kec.append(ckec)
        G.qe.append(G.compute_qe())
        G.ie.append(G.compute_ie())
        G.pe.append(G.compute_pe()) 
        
    G.estime.append(t)
    
# Function to save energy in file
def save_energy(G):
    filename = util.new_filename(Path(para.op_path)/'energies.h5')
    f = hp.File(filename, 'w')
    f.create_dataset('tenergy', data = G.te)
    f.create_dataset('t', data = G.estime)
    f.create_dataset('mu', data = G.chmpot)
    f.create_dataset('kec', data = G.kec)
    f.create_dataset('kei', data = G.kei)
    f.create_dataset('ie', data = G.ie)
    f.create_dataset('qe', data = G.qe)
    f.create_dataset('pe', data = G.pe)
    f.create_dataset('ke', data = G.ke)
    f.close()

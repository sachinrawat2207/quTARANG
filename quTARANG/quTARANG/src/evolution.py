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

from quTARANG.src.univ import fourier_transform as fft
from quTARANG.src.univ import grid
from quTARANG.src.univ import data_io as IO
from quTARANG.config.config import ncp
import time as tr
import quTARANG.config.mpara as para

#############################################################################################
#                   Numerical Schemes(without dissipation and rotation)             
#############################################################################################

###############################
# 1. TSSP Scheme
###############################

def tssp_stepr(G, dt):
    return G.wfc * ncp.exp(-1j * (G.pot + para.g  * (G.wfc * G.wfc.conj())) * dt)

def tssp_stepk(G, dt):
    return G.wfck * ncp.exp(-0.5j * grid.ksqr * dt)
    
# For real time evolution
def time_adv_strang(G):
    G.wfc[:] = tssp_stepr(G, para.dt/2)
    G.wfck[:] = fft.forward_transform(G.wfc)
    G.wfck[:] = tssp_stepk(G, para.dt)
    G.wfc[:] = fft.inverse_transform(G.wfck)
    G.wfc[:] = tssp_stepr(G, para.dt/2)

# For imaginary time evolution
def time_adv_istrang(G):
    G.wfc[:] = tssp_stepr(G, -1j * para.dt/2)
    G.wfck[:] = fft.forward_transform(G.wfc)
    G.wfck[:] = tssp_stepk(G, -1j * para.dt)
    G.wfc[:] = fft.inverse_transform(G.wfck)
    G.wfc[:] = tssp_stepr(G, -1j * para.dt/2)
    G.renorm()


###############################
# 1. RK4 Scheme
###############################     
# Need to check and validate RK4 scheme

def compute_RHS(G, psik):
    G.wfc[:] = fft.inverse_transform(psik)
    G.wfc[:] = -1j  * (grid.ksqr * psik/2 + fft.forward_transform((para.g * ncp.abs(G.wfc)**2 + G.pot) * G.wfc))
    return G.wfc

def time_adv_rk4(G):
    G.U.temp1k[:] = G.wfck + para.dt/2 * compute_RHS(G, G.wfck)
    G.U.temp2k[:] = G.wfck + para.dt/2 * compute_RHS(G, G.U.temp1k)
    G.U.temp3k[:] = G.wfck + para.dt * compute_RHS(G, G.U.temp2k)
    G.wfck[:] = G.wfck + para.dt/6 *(compute_RHS(G, G.wfck) + 2 * compute_RHS(G, G.U.temp1k) + 2 * compute_RHS(G, G.U.temp2k) + compute_RHS(G, G.U.temp3k))
    
    
#############################################################################################
#                                         Time Advance                                         
#############################################################################################

def set_scheme():
    global time_adv
    if para.scheme == 'TSSP':
        if para.imgtime == False:
            time_adv = time_adv_strang
        elif para.imgtime == True:
            time_adv = time_adv_istrang
        
    elif para.scheme == 'RK4':
        time_adv = time_adv_rk4
    else:
        print("Please choose the correct scheme")
        quit()


def time_advance(G):
    print("***** Time advence started ***** ")
    t = grid.t_initial
    if para.imgtime == True:
        img_time(G, t)
        
    elif para.imgtime == False:
        real_time(G, t)
    if para.device == 'gpu':
        para.tf_c.record()
        para.tf_c.synchronize()
        initilaization_time = ncp.cuda.get_elapsed_time(para.t0_c, para.ti_c) * 1e-3
        execution_time = ncp.cuda.get_elapsed_time(para.ti_c, para.tf_c) * 1e-3
    elif para.device == "cpu":
        para.tf_c = tr.perf_counter()
        initilaization_time = para.ti_c - para.t0_c
        execution_time = para.tf_c - para.ti_c
        
    print("-----------------------------------------")    
    print("Time taken for initialization(s): ", initilaization_time)
    print("Time take for execution(s): ", execution_time)
    elapsed_time = initilaization_time + execution_time
    print("Total run time(s): ", elapsed_time)
    print("-----------------------------------------\n")    
    print("***** Run Completed ***** ")
    
def img_time(G,t):
    E = ncp.zeros(2)
    error = 1
    i=0
    G.renorm()
    E[0] = G.compute_te()
    while error > para.delta:
        if(para.save_wfc and i >= para.wfc_start_step and (i - para.wfc_start_step)%para.wfc_iter_step == 0):
            IO.save_wfc(G, i)
        time_adv(G)
        E[1] = G.compute_te()
        error = ncp.abs(E[0]-E[1])
        E[0] = E[1]
        if i%para.t_print_step == 0:
            IO.print_params(G, error = error, energy = E[1], iter = i)
            # print(E[1], G.norm(), error)
        i+=1
        
        t += para.dt
    IO.print_params(G, error = error, energy = E[1], iter = i)
    IO.save_wfc(G,i)
        
def real_time(G, t):
    for i in range(grid.nstep):
        if(para.save_wfc and i >= para.wfc_start_step and (i - para.wfc_start_step)%para.wfc_iter_step == 0):
            IO.save_wfc(G, t)
            
        if(para.save_en and i >= para.en_start_step and (i - para.en_start_step)%para.en_iter_step == 0):
            IO.compute_energy(G, t)

        if(para.save_rms and i >= para.rms_start_step and  (i - para.rms_start_step)%para.rms_iter_step == 0):
            IO.compute_rms(G, t)
        
        if i%para.t_print_step == 0:
            IO.print_params(G, time = t)
            # print(round(t,7), G.compute_te(), G.norm())
        
        if G.potfn != None:
            G.pot[:] = G.potfn(t)                    
        
        t += para.dt
        time_adv(G)
        
    if grid.nstep > 1:
        if para.save_wfc or para.imgtime:
            IO.save_wfc(G,t)
        if (para.save_en): 
            IO.compute_energy(G, t)
            IO.save_energy(G)
            
        if (para.save_rms):
            IO.compute_rms(G, t)
            IO.save_rms(G)
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

import matplotlib.pyplot as plt
import sys 
import os
import numpy as np
import h5py as hp 
from pathlib import Path
import matplotlib.ticker as tick
import re
import imageio
from tqdm import tqdm 

# sys.path.insert(0, str(Path.cwd().parents[0]))
from src.config import *

linewidth = 2.5
plt.rcParams.update({'font.size':'20',
                     'font.family':'serif',
                     'font.weight':'bold',
                     'lines.linewidth':linewidth,
                     'text.usetex':useLatex})

class quplot2d():
    def __init__(self, ti = 0, tf = None):
        self.path = Path(path)
        sys.path.append(path)
        import para 
        self.Lx = round(para.Lx)
        self.Ly = round(para.Ly)
        self.Nx = para.Nx
        self.Ny = para.Ny
        self.ti = ti 
        self.tf = tf
        self.files = self.sortfile(self.path/'wfc')
        
    def sortfile(self, path):
        filenames =np.array(os.listdir(path))        
        # length = len(filenames)
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
        
        if self.tf != None: 
            return filearray[np.where((l<= self.tf) & (l>=self.ti))]
        
        else:
            return filearray[np.where(l>=self.ti)]
    
    def get_time(self, file_name):
        u = re.findall(r'\d+', file_name)
        t = float(u[0]+'.'+u[1])
        return t
    

    def plotpd_(self, wfc_name, path):
        fns_title = 12
        f1 = hp.File(self.path/('wfc/'+wfc_name), 'r')
        wfc_ = f1['wfc'][:]
        minwfc = 0
        maxwfc = np.max(np.abs(wfc_)**2)
        phase = np.angle(wfc_)
        f1.close()
        fig, axs = plt.subplots(1, 2, figsize = (8,3), sharey=True)
        density_ = axs[0].imshow((np.abs(wfc_)**2).T,
            origin = 'lower', extent=[-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2], cmap ='bone', vmin=minwfc, vmax=maxwfc) #'jet'
        phase_ = axs[1].imshow((phase.T), origin = 'lower', extent=[-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2],
            cmap ='plasma', vmin=-np.pi, vmax=np.pi) #'Blues'
        
        axs[0].tick_params(axis='both', which='major', labelsize=fns_title, length=2, width=1)
        axs[0].tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = fns_title)
        axs[0].set_title('Density', fontsize=fns_title)
        
        axs[1].tick_params(axis='both', which='major', labelsize=fns_title, length=2, width=1)
        axs[1].tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = fns_title)
        axs[1].set_title('Phase', fontsize=fns_title)
        frac = 0.045
        
        cbar = plt.colorbar(density_, ax=axs[0], location='right', fraction=frac, ticks=np.linspace(minwfc, maxwfc, 5))
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
        cbar.ax.tick_params(labelsize=fns_title)
        cbar = plt.colorbar(phase_, ax=axs[1], location='right', fraction=frac, ticks=np.linspace(-np.pi, np.pi, 5))
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
        cbar.ax.tick_params(labelsize=fns_title)
        time = self.get_time(wfc_name)
        name = 'denph_t%1.6f.jpeg'%time
        fig.suptitle(r"$t=$%1.3f"%time, fontsize=fns_title)
        plt.savefig(path/name, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    
    
    def plotd_(self, wfc_name, path):
        fns_title = 12
        f1 = hp.File(self.path/('wfc/'+wfc_name), 'r')
        wfc_ = f1['wfc'][:]
        minwfc = 0
        maxwfc = np.max(np.abs(wfc_)**2)
        f1.close()
        fig, ax = plt.subplots(1, 1, figsize = (4,3))
        density_ = ax.imshow((np.abs(wfc_)**2).T,
            origin = 'lower', extent=[-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2], cmap ='bone', vmin=minwfc, vmax=maxwfc) #'jet'
    
        ax.tick_params(axis='both', which='major', labelsize=fns_title, length=2, width=1)
        ax.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = fns_title)
        # ax.set_title('Density', fontsize=fns_title)
        
        frac = 0.045
        
        cbar = plt.colorbar(density_, ax=ax, location='right', fraction=frac, ticks=np.linspace(minwfc, maxwfc, 5))
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
        cbar.ax.tick_params(labelsize=fns_title)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
        cbar.ax.tick_params(labelsize=fns_title)
        time = self.get_time(wfc_name)
        name = 'den_t%1.6f.jpeg'%time
        ax.set_title(r"$t=$%1.3f"%time, fontsize=fns_title)
        # fig.suptitle(r"$t=$%1.3f"%time, fontsize=fns_title)
        plt.savefig(path/name, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    
    def plotpd(self, skip = 0):
        path = [self.path/'postprocessing/phasedensityplots']
        for dir in path:
            if not Path(dir).exists():
                os.makedirs(dir)
        l = len(self.files)
        for i in tqdm(range(0, l, skip+1)):
            wfc_name = self.files[i]
            self.plotpd_(wfc_name, path[0])
    
    def plotd(self, skip = 0):
        path = [self.path/'postprocessing/densityplots']
        for dir in path:
            if not Path(dir).exists():
                os.makedirs(dir)
        l = len(self.files)
        for i in tqdm(range(0, l, skip+1)):
            wfc_name = self.files[i]
            self.plotd_(wfc_name, path[0])

    def animation(self, fps=30, skip = 0, format = 'mp4', quality = 5, ptype = 'd'):
        dir = self.path/'postprocessing/animation'
        if not Path(dir).exists():
            os.makedirs(dir)
            
        if ptype == 'd':
            path = self.path/'postprocessing/densityplots'
        elif ptype == 'pd':
            path = self.path/'postprocessing/phasedensityplots'
            
        
        plotsname = self.sortfile(path)
        
        l = len(plotsname)
        images = []
        
        for i in tqdm(range(0, l, skip+1)):
            wfc_name = plotsname[i]
            images.append(imageio.v2.imread(path/wfc_name)) 
        if ptype == 'pd':
            kargs = { 'fps': fps, 'quality': quality, 'macro_block_size': None,'ffmpeg_params': ['-s','2064x912'] }
            loc = 'pdanim.%s'%(format)
        elif ptype == 'd':
            kargs = { 'fps': fps, 'quality': quality, 'macro_block_size': None,'ffmpeg_params': ['-s','1032x872'] }
            loc = 'danim.%s'%(format)

            
        
        imageio.mimsave(dir/loc, images, **kargs)
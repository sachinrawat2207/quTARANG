from src.config import *
import numpy as np 
import matplotlib.pyplot as plt
import h5py as hp 

linewidth = 1.2
plt.rcParams.update({'font.size':'14',
                     'font.family':'serif',
                     'font.weight':'bold',
                     'lines.linewidth':linewidth,
                     'text.usetex':useLatex})

f = hp.File(Path(path)/'energies.h5', 'r')
te = f['tenergy'][:]
ke = f['ke'][:]
kec = f['kec'][:]
kei = f['kei'][:]
kec = f['kec'][:]
qe = f['qe'][:]
pe = f['pe'][:]
t = f['t'][:]
f.close()

fig, ax = plt.subplots(1,1, figsize=(5, 4))

ax.plot(t, te, 'k-.', label = r'$E_t$')
ax.plot(t, ke, 'b-.', label = r'$E_{k}$')
ax.plot(t, kec, 'm-.', label = r'$E^{c}_{k}$')
ax.plot(t, kei, color = 'maroon', linestyle='-.', label = r'$E^{i}_{k}$')
ax.plot(t, qe,'r-.', label = r'$E_{q}$')
ax.plot(t, pe,'g-.', label = r'$E_{pe}$')

ax.set_xlim(t[0], t[-1])
ax.set_ylim(top = te[-1]+te[-1]/2)
ax.set_xlabel('$t$')
ax.set_ylabel('$E$')
ax.legend(fancybox=False,  loc='upper right', frameon=False, ncol = 2)
plt.savefig(op_dir/"energy.jpeg", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
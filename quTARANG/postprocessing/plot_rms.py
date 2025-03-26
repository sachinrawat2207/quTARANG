from src.config import *
import matplotlib.pyplot as plt

linewidth = 1.2
plt.rcParams.update({'font.size':'14',
                     'font.family':'serif',
                     'font.weight':'bold',
                     'lines.linewidth':linewidth,
                     'text.usetex':useLatex})


f = hp.File(Path(path)/'rms.h5', 'r')

if dimension == 1:
    xrms = f['xrms'][:]
    t_rms = f['t'][:]

if dimension == 2:
    rrms = f['rrms'][:]
    xrms = f['xrms'][:]
    yrms = f['yrms'][:]
    t_rms = f['t'][:]

if dimension == 3:
    xrms = f['xrms'][:]
    yrms = f['yrms'][:]
    zrms = f['zrms'][:]
    t_rms = f['t'][:]
f.close()

fig, ax = plt.subplots(1,1, figsize=(5, 4))
ax.set_ylabel("$\sigma$")
ax.set_xlabel("$t$")
ax.set_xlim(t_rms[0], t_rms[-1])

ax.plot(t_rms, xrms,'k--', label=r'$\sigma_x$', linewidth = linewidth)

if dimension == 2:
    ax.plot(t_rms, yrms,'r-.', label=r'$\sigma_y$', linewidth = linewidth)

elif dimension == 3:
    ax.plot(t_rms, yrms,'r-.', label=r'$\sigma_y$', linewidth = linewidth)
    ax.plot(t_rms, zrms, color='b', linestyle = '-.', label=r'$\sigma_z$', linewidth = linewidth)

plt.legend(fancybox=False,  loc='best', frameon=False)
plt.tight_layout()
plt.savefig(op_dir/"rms.jpeg", dpi=300, bbox_inches='tight')
plt.show()
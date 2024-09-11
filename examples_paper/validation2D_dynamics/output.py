import h5py as hp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#######################################

# Set Path to the output directory
output_dir = "/path_to_output_directory"

#######################################



file_name = Path(output_dir)/"rms.h5"
f = hp.File(file_name, 'r')
data_time_interval = np.array(f['t'])
data_obtained_xrms = np.array(f['xrms'])
data_obtained_yrms = np.array(f['yrms'])
f.close()

linewidth = 1.5
plt.rcParams.update({'font.size':'20',
                     'font.family':'serif',
                     'font.weight':'bold',
                     'lines.linewidth':linewidth,
                     'text.usetex':True})


fig, ax = plt.subplots(1,1, figsize=(5, 4))

ax.set_ylabel("$\sigma$")
ax.set_xlabel(r"$t$")
ax.set_xlim(0, 8)
ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_ylim(0.2, 1.3)
ax.plot(data_time_interval, data_obtained_xrms,'k:', label=r'$\sigma_x$', linewidth = linewidth)
ax.plot(data_time_interval, data_obtained_yrms,'r-.', label=r'$\sigma_y$', linewidth = linewidth)
ax.legend(loc=0, bbox_to_anchor=(0.58, 0.78), fontsize = 18, ncol = 2, handlelength=1.2, handletextpad=0.2, frameon=False, columnspacing=0.8)
plt.savefig("dynamics2D.jpeg", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

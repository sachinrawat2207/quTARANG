from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp
import quTARANG.config.mpara as para

##########################################################################
V = 0.5*(grid.x_mesh**2 + 4*grid.y_mesh**2)
def evolve_wfc2d():
    return 1/ncp.sqrt(ncp.sqrt(2)*ncp.pi)*ncp.exp(-(grid.x_mesh**2 + 2*grid.y_mesh**2)/4)

def evolve_pot2d(t):
    return V + 0*t

G = gpe.GPE(wfcfn = evolve_wfc2d, potfn = evolve_pot2d)
##########################################################################

evolution.time_advance(G)
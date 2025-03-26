from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp
import quTARANG.config.mpara as para

##########################################################################
V = 0.5*(grid.x_mesh**2 + 4 * grid.y_mesh**2 + 16 * grid.z_mesh**2)
def evolve_wfc3d():
    return (8/ncp.pi)**(3/4)*ncp.exp(-2*(grid.x_mesh**2 + 2 * grid.y_mesh**2 + 4 *grid.z_mesh**2))

def evolve_pot3d(t):
    return V + 0*t

G = gpe.GPE(wfcfn = evolve_wfc3d, potfn = evolve_pot3d)
##########################################################################

evolution.time_advance(G)
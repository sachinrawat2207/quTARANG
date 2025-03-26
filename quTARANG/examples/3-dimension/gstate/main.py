from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp
import quTARANG.config.mpara as para

##########################################################################
V = 0.5*(grid.x_mesh**2 + grid.y_mesh**2 + 4**2 * grid.z_mesh**2)
def gstate_wfc3d():
    return 1/(ncp.pi)**(3/4)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2+ grid.z_mesh**2)/2)

def gstate_pot3d(t):
    return V + 0*t

G = gpe.GPE(wfcfn = gstate_wfc3d, potfn = gstate_pot3d)
##########################################################################

evolution.time_advance(G)

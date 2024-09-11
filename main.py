from quTARANG import para
from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp

##########################################################################
# V = 0.5*(grid.x_mesh**2+grid.y_mesh**2)

# def wfc():
#     return 1/ncp.sqrt(ncp.pi)*ncp.exp(-(grid.x_mesh**2+grid.y_mesh**2)/2) 

# def pot(t): 
#     return V


### Initial condtition chosen for runs in paper
# gstate2d
def gstate_wfc2d():
    return 1/ncp.sqrt(ncp.pi)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2)/2)

def gstate_pot2d(t):
    return 0.5*(grid.x_mesh**2 + grid.y_mesh**2)

# gstate3d
def gstate_wfc3d():
    return 1/(ncp.pi)**(3/4)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2+ grid.z_mesh**2)/2)

def gstate_pot3d(t):
    return 0.5*(grid.x_mesh**2 + grid.y_mesh**2 + 4**2 * grid.z_mesh**2)

# evolve2d
def evolve_wfc2d():
    return 2**(1/4)/ncp.sqrt(2*ncp.pi)*ncp.exp(-(grid.x_mesh**2 + 2*grid.y_mesh**2)/4)

def evolve_pot2d(t):
    return 0.5*(grid.x_mesh**2 + 4*grid.y_mesh**2)

# evolve3d
def evolve_wfc3d():
    return (8/ncp.pi)**(3/4)*ncp.exp(-2*(grid.x_mesh**2 + 2 * grid.y_mesh**2 + 4 *grid.z_mesh**2))

def evolve_pot3d(t):
    return 0.5*(grid.x_mesh**2 + 4 * grid.y_mesh**2 + 16 * grid.z_mesh**2)

G = gpe.GPE(wfcfn = None, potfn = None)
##########################################################################

evolution.time_advance(G)
from quTARANG import para
from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp

##########################################################################
G = gpe.GPE()
##########################################################################

evolution.time_advance(G)
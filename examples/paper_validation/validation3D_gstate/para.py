#================================================================================
#                       Change the following parameters
#================================================================================
real_dtype = "float64"
complex_dtype = "complex128"

pi = 3.141592653589793

# Device Setting 
device = "gpu"             # Choose the device <"cpu"> to run on cpu and gpu to run on <"gpu">
device_rank = 1            # Set GPU no in case if you are running on a single GPU else leave it as it is

# Set grid size 
Nx = 256
Ny = 256
Nz = 256
    
# Set box length
Lx = 32
Ly = 32
Lz = 32

# Set maximum time and dt
tmax = 1    
dt = 0.001

# Choose the value of the non linerarity
g = 18.81

init_usrdef = True
init_cond = "rp"

# If init_usrdef is True then either pass input through main or set the input path along with input file
in_path = "/path/to/input"


# Set output folder path
op_path = "../output_gstate3D"

# Choose the scheme need to implement in the code
scheme = "TSSP"          # Choose the shemes <"TSSP">, <"RK4"> etc

imgtime = True          # set <False> for real time evolution and <True> for imaginary time evolution
delta = 1e-12


# To resume the Run
resume = False

overwrite = True

# Wavefunction save setting
wfc_start_step = 0

# make wfc_iter too big to stop saving the wfc 
wfc_iter_step = 500

# Rms save setting
save_rms = True
rms_start_step = 0
rms_iter_step = 10

# Energy save setting
save_en = True
en_start_step = 0
en_iter_step = 100

# Printing iteration step
t_print_step = 1000


#================================================================================
if Nx != 1 and Ny == 1 and Nz == 1:
    dimension = 1

elif Nx != 1 and Ny != 1 and Nz == 1:
    dimension = 2

elif Nx != 1 and Ny != 1 and Nz != 1:
    dimension = 3  
#================================================================================

# Used for computation of time
t0_c = 0
ti_c = 0
tf_c = 0

#================================================================================
#                       Change the following parameters
#================================================================================
real_dtype = "float64"
complex_dtype = "complex128"

pi = 3.141592653589793

# Device Setting 
device = "gpu"             # Choose the device <"cpu"> to run on cpu and gpu to run on <"gpu">
device_rank = 0            # Set GPU no in case if you are running on a single GPU else leave it as it is

# Set grid size 
Nx = 256
Ny = 256
Nz = 1

# Set box length
Lx = 29.82
Ly = 29.82
Lz = 1

# Set maximum time and dt
tmax = 600
dt = 0.001

# Choose the value of the non linerarity
g = 4

inp_type = "pdf"       # Choose the initial condition type <"fun">, <"dat">, <"pdf">

typ = "rp"            # In case of inp_type = "pdf" set the type of initial condition <"rp">, <"rv">, <"vl"> for 2D and <"rp"> for 3D.

# If inp_type = "dat" then set the input path
in_path = "/path/to/input_directory"

# Set output folder path
op_path = "../OUTPUTDATA"

scheme = "TSSP"         

imgtime = False          # set <False> for real time evolution and <True> for imaginary time evolution
delta = 1e-12

overwrite = False

# Wavefunction save 
save_wfc = True
wfc_start_step = 0
wfc_iter_step = 5000

# Rms save setting
save_rms = False
rms_start_step = 0
rms_iter_step = 10

# Energy save setting
save_en = True
en_start_step = 0
en_iter_step = 100

# Printing iteration step
t_print_step = 5000

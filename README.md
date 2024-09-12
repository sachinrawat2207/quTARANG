[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**quTARANG** is a Python package designed for studying turbulence in quantum systems, specifically in atomic Bose-Einstein condensates (BECs), using the mean-field Gross-Pitaevskii equation (GPE) given by

$$
i\hbar\partial_t\psi(\vec{r},t) = -\frac{\hbar^2}{2m}\nabla^2\psi(\vec{r},t) + V(\vec{r},t)\psi(\vec{r},t) + U_0|\psi(\vec{r},t)|^2\psi(\vec{r},t),
$$

where $\psi(\vec{r},t)$ is the macroscopic complex wave function, $m$ is the atomic mass, $V(\vec{r},t)$ is the trapping potential, $N$ is the number of particles, $\displaystyle U_0=(4\pi\hslash^2a_s)/m$ is the nonlinear interaction parameter and $a_s$ denotes the scattering length for the interaction of the atomic particles.

The package is hardware-agnostic, allowing users to run simulations on either a CPU or a GPU by setting the `device` parameter in the `para.py` file, as detailed below. **quTARANG** employs the Time-Splitting Pseudo-Spectral (TSSP) method for evolving the system, ensuring both efficiency and accuracy. Additionally, the package can compute stationary states by evolving the GPE in imaginary time, enabled by setting `imgtime` parameter as `True` in the para.py file. In addition, **quTARANG** also includes functions to compute various statistical quantities like spectra and fluxes. It can compute the energy spectra by using a conventional binning method as well as more resolved spectra using the the angle-averaged Wiener-Khinchin approach [see](https://journals.aps.org/pra/pdf/10.1103/PhysRevA.106.043322). 

The directory structure of **quTARANG** package is as follows:
```
├── postprocessing
└── quTARANG
    ├── config
    ├── initial_cond
    ├── util
    ├── src
    └── para.py
├── main.py
```
- `postprocessing` directory contains functions to plot the energy evolution, function to compute the spectra and fluxes of compressible kinetic energy, incompressible kinetic energy and particle number, once the run for a simulation completes.

- `postprocessing` directory contains functions that allow user to plot the energy evolution and compute the spectra and fluxes of compressible kinetic energy, incompressible kinetic energy, and particle number,once a simulation run completes.

- `quTARANG` directory contains the source files and a `para.py` file, where a user needs to set the parameters required to perform a sumulaition.

- `main.py` is used to provide user defined intial condition as well as to run the code, once all the necessary input parameters are set.

## 1) Packages required to run quTARANG
The following Python packages need to be installed to run quTARANG
* `numpy` : To run the code on a CPU,
* `cupy` : To run the code on a GPU,
* `h5py`,
* `matplotlib`,
* `pyfftw`.


## 2) Steps to run the code
1. Set the parameters inside the following `para.py` file:
    ```python
    #para.py
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
    Nx = 512
    Ny = 512
    Nz = 1
        
    # Set box length
    Lx = 22
    Ly = 22
    Lz = 1

    # Set maximum time and dt
    tmax = 8    
    dt = 0.001

    # Choose the value of the non linerarity
    g = 0.1

    init_usrdef = False
    init_cond = 'rp'

    # If init_usrdef is True then either pass input through main or set the input path along with input file
    in_path = "/path/to/input/directory/"


    # Set output directory path
    op_path = "/path/to/output/directory/"

    # Choose the scheme need to implement in the code
    scheme = "TSSP"          # Choose the shemes <"TSSP">, <"RK4"> etc
    imgtime = False          # set <False> for real time evolution and <True> for imaginary time evolution

    delta = 1e-12 # Soping criteria in imaginary time evolution 

    # To resume the Run
    resume = False

    # To overwrite the existing output
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
    ```

The description of the parmeters inside the para.py file is as follows: 

* `real_dtype` and `complex_dtype` : These prameter set the precison of the arrays used in the code. `real_dtype` specifies the precision for real array and it can takes two values *`"float32"`* or *`"float64"`* for single and double precision arrays, while `complex_dtype` can takes the values *`"complex64"`* or *`"complex128"`* for setting the complex array as single and double precision, respectively.
* `device` : Specify the device on which the code will run. Set it to *`"cpu"`* or *`"gpu"`* to run the code on the central processing unit (CPU) or graphics processing unit (GPU), respectively.
* `device_rank` : For systems with multiple GPUs, this parameter specifies the GPU device on which the code will run. It accepts values from 0 (corresponding to the first GPU) to (number of GPUs - 1). The default value is 0.
* `(Lx, Ly, Lz)` and `(Nx, Ny, Nz)` : These parameters define the box lengths and grid sizes along the $x-, \ y-$, and $z-$ directions. In two-dimensional simulations (2D runs), set `Lz` (length in z-direction) and `Nz` (number of grid points in z-direction) to 1.

* `tamx, dt` :The `tmax` parameter specifies the total simulation time. It defines the maximum duration for which the simulation will run. The `dt` parameter, on the other hand, determines the time step size. 

* `g` : Sets the value of nonlinearity in the system.
* `init_usrdef` : This parameter controls whether the initial condition is defined by the user or by the code itself. It accepts Boolean values: *`True`* for user-defined initial conditions and *`False`* for code-defined initial conditions. The user can define user-defined initial conditions in two ways.
    - The first way is to provide the path of the input directory to the parameter `in_path` containing initial wavefunction and potential data in the form of HDF5 files named `wfc.h5` and `pot.h5`. The datasets inside the `wfc.h5` and `pot.h5` files should be named `wfc` and `pot`, respectively. 
    - The second way of defining the user-defined initial condition is through the `main.py` file which is described [here](#mainpy).  

    For the *`False`* values of `init_usrdef` i.e., for code defined initial condition, the user needs to set the `init_cond` parameter.

* `init_cond` :  If `init_usrdef` parameter is *`False`*, it will set the type of the code defined initial condition. In 2D simulations, it can take the values *`"rp"`* (random phase), *`"vl"`* (vortex lattice), and *`"rv"`* (random vortex). The number of vortices for *`"vl"`* and *`"rv"`* is specified in the quTARANG/initial_cond/dimension2.py file can be change accordingly inside the same file. For 3D simulations, only the *`"rp"`* (random phase) option is available. If init_usrdef is set to *`True`*, leave it unchanged.

* `in_path` : Sets the path to the input directory containing the initial wavefunction and potential data if `init_usrdef` is *`True`* and the wants to provide initial condition through input files. 

* `op_path` : Sets the path for the output directory where the data of the output will be stored.
* `scheme` : Sets the scheme for the simulation. Currently, only the Time-splitting pseudo-spectral (TSSP) scheme is implemented in the code. Therefore, the parameter can only take the value *`"TSSP"`*.
* `imgtime` : To compute the ground state, set this parameter to *`True`*. The code will then utilize the imaginary time approach to find the ground state.
* `delta` : This parameter will sets the stopping criteria in case of computing the ground state. The deafault value of this parameter is *`1e-12`*.
* `resume` : If the user wants to extend a simulation run, the user can do so by increasing the value of the `tmax` parameter and setting `resume` parameter to *`True`*.  The code will automatically load the most recently saved wavefunction and resume the simulation from that point.
* `overwrite` : If the output directory already contains data and the user wants to overwrite it, he can do so by simply setting this parameter to *`True`*.
* `wfc_start_step, wfc_iter_step` :  The `wfc_start_step` parameter determines the number of iterations after which the wavefunction starts saving. The `wfc_iter_step` parameter then controls the interval between subsequent wavefunction saves.
* `save_rms, rms_start_step, rms_iter_step` : The `save_rms` parameter controls whether the code saves the root mean square (RMS) value of the condensate. Set it to *`True`* to enable saving RMS values otherwise set it to *`False`*. If user choose to enable RMS data saving, the `rms_start_step` and `rms_iter_step` parameter comes into play. `rms_start_step` parameter determines the number of iterations after which the RMS values are first saved and then `rms_iter_step` controls the interval between subsequent saves.

* `save_en, en_start_step, en_iter_step` : Similarly, the `save_en` parameter controls whether the code saves energy data, while `en_start_step` and `en_iter_step` determine the starting iteration and interval for subsequent energy saves, respectively.
* `t_print_step` : Sets the interval of no of after which the code will print on the terminal.



2. Modify the following `main.py` file: 
    <a id="mainpy"></a>
    The user can define the functions that generate the initial wavefunction and potential through the `main.py`. These functions require three inputs:
    - `ncp`: An instance of either NumPy or cuPy, depending on the device used for code execution.
    - `grid`: An object containing the grids for the $x-, \ y-$, and $z-$ axes, stored in the `x_mesh`, `y_mesh`, and `z_mesh` variables, respectively.

   The potential function is time dependent potential and takes a input parameter *`t`* and user can defines potential fuction which is depending on the time. Once the functions are defined the user needs to change the instance of the `gpe` class as `G = gpe.GPE(wfcfn = wfc_func, potfn = pot_func)`, where `wfc_func` and `pot_func` are the user defined functions for inital wavefunction and potential respectively.  

   If user don't want to pass input through the `main.py` file, he can set the instance of the `gpe` class to `G = gpe.GPE(wfcfn = None, potfn = None)`.

   Following is the example code in `main.py` file.

    ```python
    #main.py
    from quTARANG import para
    from quTARANG.src.lib import gpe
    from quTARANG.src import evolution
    from quTARANG.src.univ import grid, fns
    from quTARANG.config.config import ncp

    # 2D user defined initial conditon 
    def gstate_wfc2d():
        return 1/ncp.sqrt(ncp.pi)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2)/2)

    def gstate_pot2d(t):
        return 0.5*(grid.x_mesh**2 + grid.y_mesh**2)

    # 3D user defined intial conditon
    def gstate_wfc3d():
        return 1/(ncp.pi)**(3/4)*ncp.exp(-(grid.x_mesh**2 + grid.y_mesh**2+ grid.z_mesh**2)/2)

    def gstate_pot3d(t):
        return 0.5*(grid.x_mesh**2 + grid.y_mesh**2 + 4**2 * grid.z_mesh**2)


    G = gpe.GPE(wfcfn = None, potfn = None)

    # If init_usrdef = True in para.py file and user want to pass the user defined funcitns.
    # G = gpe.GPE(wfcfn = gstate_wfc2d, potfn = gstate_pot2d)

    ##########################################################################

    evolution.time_advance(G)
    ```
3. Once the `para.py` and `main.py` file sets, the one can run the simulations using:
    ```python
    python3 main.py
    ```

## 3) Ouputs
When the simulation completes successfully, the following directories/files are generated inside the output directory:
- `wfc` : The `wfc` directory stores wavefunctions at different points in time. The filenames follow the format `wfc_<time>.h5`, where `<time>` represents the simulation time at which the wavefunction was saved. For example, `wfc_10.000000.h5` indicates that the wavefunction was saved at time $t = 10$.
- `energies.h5` : This file contains the different types of energies saved at the times mentioned in the `para.py` file. This file will only be generated if the `save_en` parameter within `para.py` file is set to *`True`*.
- `rms.h5` : Similaraly, this files contains the RMS values: $x_{rms}, \ y_{rms}$, $r_{rms}$ for a 2-D run, and $x_{rms}, \ y_{rms} \ z_{rms}$, $r_{rms}$ for a 3-D runs. Like `energies.h5` file, this file will only be generated if `save_rms` parameter within `para.py` file is set to *`True`*.

- `para.py` and `main.py`: These are copies of the original parameter and main program files (`para.py` and `main.py`) used at the time the simulation was run. These captured files allow user to reproduce the exact simulation conditions used.


## 4) Postprocessing 
Once the output from ***quTARANG*** has been generated in the output directory, users can post-process the data using the functions within the `postprocessing` directory. The structure of the directories and files within the `postprocessing` directory is as follows:

```
├── plot_energy.py
├── plot_rms.py
├── plot_spectra.ipynb
└── postprocessing
    ├── config.py
    └── libs
```

The main file within the `postprocessing` directory is `config.py`. Here, users set the location of the output data directory. After setting this location, users can plot energy and RMS evolution by running `plot_energy.py` and `plot_rms.py`, respectively. The output plots will be stored in a newly created subdirectory named `postprocessing` within the original output directory. Finally, the Jupyter Notebook `plot_spectra.ipynb` contains the commands to generate spectra and fluxes, along with comments explaining their purpose.

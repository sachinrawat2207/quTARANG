[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


**quTARANG** is a Python package designed for studying turbulence in quantum systems, specifically in atomic Bose-Einstein condensates (BECs). It utilizes the mean-field Gross-Pitaevskii equation (GPE) to simulate the dynamics of BECs. The non-dimensional GPE implemented in quTARANG is given by

$$
i\partial_t\psi(\vec{r},t) = -\frac{1}{2}\nabla^2\psi(\vec{r},t) + V(\vec{r},t)\psi(\vec{r},t) + g|\psi(\vec{r},t)|^2\psi(\vec{r},t),
$$

where $\psi(\vec{r},t)$ is the  non-dimensionalized macroscopic complex wave function, $V(\vec{r},t)$ is the non-dimensionalized external potential, and $g$ is the coefficient of non-linearity, governing the strenth of interactions within the system.

This package is hardware-agnostic, allowing users to run simulations on either a CPU or a GPU by simply changing a flag. **quTARANG** uses the Time-Splitting Pseudo-Spectral (TSSP) method for evolving the system, ensuring both efficiency and accuracy. Additionally, the package can compute stationary states by evolving the GPE in imaginary time. It is equipped with the functions to compute statistical quantities like spectra and fluxes. It can compute the energy spectra using a conventional binning method, and resolved spectra using the angle-averaged Wiener-Khinchin approach [see](https://journals.aps.org/pra/pdf/10.1103/PhysRevA.106.043322). 

## Directory structure of quTARANG
The directory structure of **quTARANG** package is as follows:
```
├── quTARANG
    ├── config
    ├── initial_cond
    ├── util
    ├── src
├── examples
├── para.py
├── main.py
└── postprocessing
```
- `quTARANG` directory contains quTARANG's source files.
- `examples` directory contains some working examples of different 2-D and 3-D cases for both stationary state computation and dynamical evolution of the system.
- `para.py` contains different simulation parameters.
- `main.py` is the file that should be executed to start the simulation The initial condition for the simulation can be set in the `main.py` file. There are other ways to set the initial condition, which have been described in the **Setting Initial Conditon** section.
- `postprocessing` directory contains libraries and files used for data postprocessing and visualization, which includes:
    1. Computation and plotting of the spectra of various quantities of interest such as incompressible kinetic energy, compressible kinetic energy and particle number using either the conventional binning method or using the angle-averaged Wiener-Khinchin approach.
    2. Computation and plotting of fluxes for incompressible kinetic energy, compressible kinetic energy and particle number.
    3. Ploting the time series of the total energy and its various components as well as the root mean square (rms) size of the condensate along different directions.

## Packages required to run quTARANG
The following Python packages must be installed to run quTARANG and for the postprocessing of the data generated during the simulation:

    * `numpy` : To run the code on a CPU,
    * `cupy` : To run the code on a GPU,
    * `h5py` : To save the output in HDF5 format,
    * `matplotlib` : For data visualization,
    * `pyfftw`: To perform FFT on CPU devices,
    * `imageio, imageio[ffmpeg]` : To generate animations from the simulation data
    * `scipy`
    * `tqdm` : To display the progress bar during the animation generation.

The above packages can be installed using the following command:    
```python
pip3 install numpy h5py matplotlib pyfftw imageio scipy imageio[ffmpeg] tqdm
```
For installation of `cupy`, the user can follow the instructions provided in the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

In addition to the above packages, the user can also install LaTeX (preferably TeX-Live) to generate symbols and numbers in the LaTeX style while generating plots using quTARANG. The `cm-super, dvipng, texlive-latex-extra`, and `texlive-fonts-recommended` packages are also required for latex to work in `matplotlib` and can be installed using the following command:
```bash
sudo apt-get install texlive-full dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```



## Running quTARANG
In order to run quTARANG, the user needs to configure the parameters and set the initial conditions for the simulation. This is done using the para.py and main.py files respectively. Some working examples are provided in the `example` directory, the user can copy the contents of the `main.py` and `para.py` files from the `example` directory to  `main.py` and `para.py` files of the `quTARANG` directory. Once the `para.py` and `main.py` file have been set, the simulation can be executed using the followng command:
```python
python3 main.py
```

## Description of `para.py` and `main.py` file.

The parameters within the `para.py` file are described as follows:

| Parameters | Description | Values |
|------------|-------------|--------|
| `real_dtype` | Sets the precision of the real arrays used in the code. | `"float32"`: Single precision. <br> `"float64"`: Double precision. |
| `complex_dtype` | Sets the precision of the complex arrays used in the code. | `"complex32"`: Single precision. <br> `"complex64"`: Double precision. |
| `device` | Sets the device on which the code will run. | `"cpu"`: Runs the code on a CPU. <br> `"gpu"`: Runs the code on a GPU. |
| `device_rank` | Sets which GPU to use in multi-GPU systems, with values ranging from 0 to `(number of GPUs - 1)`. Default is 0. Remains ineffective for `device = "cpu"`. | `0` to `(number of GPUs - 1)`. |
| `Nx, Ny, Nz` | Sets the grid sizes along the $x$-, $y$-, and $z$-axes. | Set `Ny=1`, `Nz=1` for 1-D and `Nz=1` for 2-D simulations.  `Nx`, `Ny`, and `Nz` can take only values of dtype Integer. |
| `Lx, Ly, Lz` | Sets the box lengths along the $x$-, $y$-, and $z$-axes. | Set `Ly=1`, `Lz=1` for 1-D and `Lz=1` for 2-D simulations. |
| `tmax, dt` | `tmax` sets the total simulation time, while `dt` determines the time step of the simulation. | |
| `g` | Sets the value of the nonlinearity parameter for the system. | |
| `inp_type` | This parameter determines how the initial condition will be set in quTARANG. | `"fun"`: The initial condition is defined through a function (to be specified in `main.py` file). <br><br> `"dat"`: Initial conditions are set using `HDF5` files, one each for the wavefunction and the potential of the system. The structure of the HDF5 file needed for this purpose has been described in **Setting Initial Conditions/Case II** section. <br><br> `"pdf"`: Stands for "Predfined Functions". The initial condition is set using functions that have been predefined in `quTARANG`. The parameter `typ`, described below, can be used to set one function among the various available options. |
|`typ`|When `inp_type="pdf"`, this parameter sets the type of initial condition. Remains ineffective for other values of `inp_type`.|2D: `"rp"`, `"rv"`, and `"vl"` correspond to smooth random phase, random vortices, and vortex lattice initial conditions, respectively.<br><br> 3D: `"rp"` corresponds to a smooth random phase initial condition.|
| `in_path` | Sets the directory containing the `HDF5` files for the initial wavefunction and potential when `inp_type = "dat"`. Remains ineffective for other value of `inp_type`. | `"path/to/input/directory"` |
| `op_path` | Sets the output directory, where simulation data generated during the simulation will be saved. | `"path/to/output/directory"` |
| `scheme` | Sets the numerical scheme of the simulation. In the current version, only TSSP is supported. Other numerical schemes will be added in future updates. | `"TSSP"` |
| `imgtime` |  Sets whether the code computes the stationary state or evolves the system in real time. | `True`: Compute stationary state,<br> `False`: Evolve the system in real time. |
| `delta` | Stopping criteria for stationary state computation (`imgtime = True`). It is the absolute difference in the total energy of the wavefunction between consecutive time steps, i.e., $\delta = \|E_n - E_{n-1}\|$. Remains ineffective for `imgtime = False`. | |
| `overwrite` | Prevents accidental overwriting in case there is data already present inside the provided output directory. | `True` (overwrite), `False` (do not overwrite). |
|`save_wfc`| Sets whether the wavefunctions will be saved or not. |`True` (save), `False` (do not save).|
| `wfc_start_step`, `wfc_iter_step` | `wfc_start_step`: The number of iterations after which the wavefunction starts saving. `wfc_iter_step`: The interval between subsequent wavefunction saves. Remains ineffective for `save_wfc = False`. | |
| `save_rms` | Sets whether to save the time series of the root mean square (rms) size of the condensate. | `True` (save), `False` (do not save). |
| `rms_start_step, rms_iter_step` | Controls rms size saving behavior. Similar to `wfc_start_step, wfc_iter_step`. | |
| `save_en` | Sets whether to save the time series of the total energy and its various components. | `True` (save), `False` (do not save). |
| `en_start_step, en_iter_step` | Controls energy saving behavior. Similar to `wfc_start_step, wfc_iter_step`. | |
| `t_print_step` | Sets the interval after which an update on the simulation status will be printed on the terminal. | |

### Setting Initial Conditon
Based on the type of input for initial condition (`inp_type="fun"`, `inp_type="dat"` or `inp_type="pdf"`), the following are the possible cases for setting up the initial condition: 
#### Case I: Setting up the initial condition by using functions 
For `inp_type = "fun"` in `para.py`, the user has to define the functions for the initial wavefunction and potential in `main.py`. These functions can be defined using the following aliases:

- `ncp`: An alias of NumPy (for a CPU) or CuPy (for a GPU), depending on the device used for the code execution.
- `grid`: An alias of quTARANG's grid module containing the grids for the $x-, \ y-$, and $z-$ axes, and can be accessed using `x_mesh`, `y_mesh`, and `z_mesh` variables, respectively.

The following is an example showing the `main.py` file, where the functions are defined for the wavefunction, $\psi(\vec{r},0)=\left(\frac{1}{\sqrt{2}\pi}\right)^{1/2} e^{-(x^2+2y^2)/4}$ and potential, $V(\vec{r},0)=\frac{1}{2}(x^2+4y^2)$.

```python
#main.py
from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp
import quTARANG.config.mpara as para

##########################################################################
V = 0.5*(grid.x_mesh**2 + 4*grid.y_mesh**2)
def wfcfn():
    return 1/ncp.sqrt(ncp.sqrt(2)*ncp.pi)*ncp.exp(-(grid.x_mesh**2 + 2*grid.y_mesh**2)/4)

def potfn(t):
    return V + 0*t

G = gpe.GPE(wfcfn = wfcfn, potfn = potfn)
##########################################################################

evolution.time_advance(G)
```
The potential function can be time-dependent so an input parameter `t` needs to be present in its definition even if the potential for a particular case is time-independent. The user can define a time dependent potential fuction using that `t`. Once the functions are defined, the user needs to pass references of those functons to `G`, an instance of the `gpe` class as `G = gpe.GPE(wfcfn = wfc_func, potfn = pot_func)`, where `wfc_func` and `pot_func` are reference to the defined functions.

#### Case II: Setting initial condtion using predefined initial condition or  by passing the path of the directory containing the wavefunction and potential files in `HDF5` format.
For both `inp_type="dat"` and `inp_type="pdf"` in `para.py`, the `main.py` will remain same. When `inp_type = "dat"`, the user has to set `inp_path` variable of `para.py` to the path of the directory containing the `HDF5` files for the wavefunction and potential. The files configured in the following way : (1) `wfc.h5` should have a dataset named `wfc` containing the wavefunction data, and (2)`pot.h5` should have a dataset named `pot` containing the potential data. 

While for `inp_type = "pdf"`, the user has to set the `typ` variable which can take the following values:
2D: `"rp"`, `"rv"`, and `"vl"` correspond to smooth random phase, random vortices, and vortex lattice initial conditions, respectively. 
3D: `"rp"` corresponds to a smooth random phase initial condition.
More information regarding these intial conditions can be found in the  `readme.md` files of examples corresponding to each case in the `example` directory.

The following is the `main.py` file corresponding to this case. 
```python
#main.py
from quTARANG.src.lib import gpe
from quTARANG.src import evolution
from quTARANG.src.univ import grid, fns
from quTARANG.config.config import ncp
import quTARANG.config.mpara as para

##########################################################################
G = gpe.GPE()
##########################################################################

evolution.time_advance(G)
```


## Ouputs
When the simulation completes successfully, the output files generated during runtime will be saved inside the output directory. These files are stored in `HDF5` format. In case of dynamical evolution (`imgtime = False` in `para.py`), the following directories/files are generated inside the output directory.:
- `wfc` : The `wfc` directory stores wavefunctions at different points in time. The filenames follow the format `wfc_<time>.h5`, where `<time>` represents the simulation time at which the wavefunction was generated. For example, `wfc_10.000000.h5` indicates that the wavefunction was generated at time $t = 10$. In each of the wavefunction file, the data is saved in the `wfc` dataset which can be easily accessed by Python's `h5py` library.
- `pot.h5` : This file contains the data for the potential at `t=0` in a `pot` dataset inside. 
- `energies.h5` : This file contains the time series of different types of energies.
- `rms.h5` : This files contains the time series of rms size of the condensate along different axes: $x_{rms}, \ y_{rms}$, $r_{rms}$ for a 2-D run, and $x_{rms}, \ y_{rms} \ z_{rms}$, $r_{rms}$ for a 3-D run.

- `para.py` and `main.py`: These are copies of the original parameter and main files (`para.py` and `main.py`) used at the time of the simulation. These files allow user to check the initial conditions and parameters used for the simulation.

In case of computing the stationary state (`imgtime = True` in `para.py`), the wavefunctions at different iterations and after the end of all iterations will be saved in the `wfc` directory inside the `ouput` directory. If `save_wfc=False` in `para.py` file, it will only generate one wavefuncton file at the end of all iterations. The wavefuntion filenames follow the format `wfc_<iterations>.h5`, where `<iterations>` represents the no. of iterations after which the wavefunction was saved. For example, `wfc_1000.h5` indicates that the wavefunction was saved at the end of the 1000th iteration. The wavefuntion whoose iteration number is largest among all the wavefunction will be the wavefunction corresponding to the ground state of the system. In each wavefunction file, the data will be stored in the `wfc` dataset.

## Postprocessing 
Once a simulation using ***quTARANG*** has completed successfully, the data generated in the output directory can be postprocessed using the python scripts within the `postprocessing` directory. The structure of the directories and files within this directory are as follows:

```
├── src
├── op_path.py    
├── plot_energy.py
├── plot_rms.py
├── plot_animation.ipynb
└── plot_spectra.ipynb
``` 

The `src` directory contains the classes and functions for the computation of spectra, fluxes and generation of animation. The user need not change the content of the src file. 

To perform postprocessing on the data, the user has to simply set the path of the data which was generated by quTARANG in the `op_path.py` file. Once the path has been set, a user can plot the time series for the energy and rms size by simply executing the scripts `plot_energy.py` and `plot_rms.py`, respectively. The plots and animations correspoending to the density and phase can be generated using the jupyter notebook named `plot_animation.ipynb` while the spectra and flux plots can be generated using another jupyter notebook named `plot_spectra.ipynb`. The comments on the cells of these notebooks explains the usage of the code inside those cells. The plots generated will be saved inside a newly automatically created subdirectory named `postprocessing` within the output directory.

## Test cases

1. 2D case:  

    $$\psi(\vec{r},0)=\left(\frac{1}{\sqrt{2}\pi}\right)^{1/2} e^{-(x^2+2y^2)/4}$$

    $$V(\vec{r},0)=\frac{1}{2}(x^2+4y^2)$$ 

    The corresponding `main.py` and `para.py` files for this case are as follows:

    ```python
    #main.py
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
    ```

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
    device_rank = 1            # Set GPU no in case if you are running on a single GPU else leave it as it is

    # Set grid size 
    Nx = 256
    Ny = 256
    Nz = 1
        
    # Set box length
    Lx = 16
    Ly = 16
    Lz = 1

    # Set maximum time and dt
    tmax = 8    
    dt = 0.001

    # Choose the value of the non linerarity
    g = 2

    inp_type = "fun"       # Choose the initial condition type among <"fun">, <"dat"> and <"pdf">

    typ = "rp"            # In case of inp_type = "pdf" set the type of initial condition <"rp">, <"rv">, <"vl"> for 2D and <"rp"> for 3D.

    # If inp_type = "dat" then set the input path
    in_path = "/path/to/input_directory"


    # Set output folder path
    op_path = "../output_evolve2D"

    scheme = "TSSP"         

    imgtime = False          # set <False> for real time evolution and <True> for imaginary time evolution
    delta = 1e-12

    overwrite = False

    # Wavefunction save setting
    save_wfc = True
    wfc_start_step = 0
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


    ```
    On the successful execution of the code, a user can postprocess the data by first setting the `path` variable in the `op_path.py` file located in the `postprocessing` directory, after which the user can perform the analysis.

    The following plots show the time series of the rms size and energy of the condensate generated by running `plot_rms.py` and `plot_energy.py`:

    | ![Image 1](img_readme/2D/rms.jpeg) | ![Image 2](img_readme/2D/energy.jpeg) |
    |:---------------------:|:----------------------:|

2. 3D case:

    $$\psi(\vec{r},0)=\left(\frac{8}{\pi}\right)^{3/4}\text{exp}(-2x^2-4y^2-8z^2)$$

    $$V(\vec{r},0)=\frac{1}{2}(x^2+4y^2+16z^2)$$

    The corresponding `main.py` and `para.py` files for this case are as follows: 

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
    Nx = 256
    Ny = 256
    Nz = 256
        
    # Set box length
    Lx = 16
    Ly = 16
    Lz = 16

    # Set maximum time and dt
    tmax = 5    
    dt = 0.001

    # Choose the value of the non linerarity
    g = 0.1

    inp_type = "fun"        # Choose the initial condition type among <"fun">, <"dat"> and <"pdf">

    typ = "rp"            # In case of inp_type = "pdf" set the type of initial condition <"rp">, <"rv">, <"vl"> for 2D and <"rp"> for 3D.

    scheme = "TSSP"
    # If inp_type = "dat" then set the input path
    in_path = "/path/to/input_directory"

    # Set output folder path
    op_path = "../output_evolve3D"


    imgtime = False          # set <False> for real time evolution and <True> for imaginary time evolution
    delta = 1e-12

    overwrite = False

    # Wavefunction save setting
    save_wfc = True
    wfc_start_step = 0
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

    ```
    ```python
    #main.py
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
    ```

    The following plots show the time series of the rms size and energy of the condensate generated by running `plot_rms.py` and `plot_energy.py`:

    | ![Image 1](img_readme/3D/rms.jpeg) | ![Image 2](img_readme/3D/energy.jpeg) |
    |:---------------------:|:----------------------:|


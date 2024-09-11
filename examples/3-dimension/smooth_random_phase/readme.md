The following initial conditions are set using `main.py` and `para.py`:

$$V(\vec{r},0) = 0,$$


$$\psi(\vec{r},0)=e^{i\theta(\vec{r}, 0)},$$

with

$$\theta'(\vec{k},0)=\begin{cases}
      \theta_0 e^{i\alpha(\vec{k})} &\delta k\leq|\vec{k}|\leq3\delta k,\\ 
      0 & \text{otherwise},
     \end{cases}$$
       
where $\theta'(\vec{k},0)$ is the Fourier transform of $\theta(\vec{r},0)$, $\theta'(\vec{k},0)$ =  $\theta'^*(-\vec{k},0)$, $\alpha(\vec{k})$ is chosen randomly for each $\vec{k}$ in interval $[-\pi, \pi]$ and $\delta k = 2\pi/L$. 

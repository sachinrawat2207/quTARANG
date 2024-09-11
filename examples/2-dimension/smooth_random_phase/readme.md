The following initial conditions are set using `main.py` and `para.py`:

$V(\bm{r},0) = 0,$
$\psi(\bm{r},0)=e^{i\theta(\bm{r}, 0)}$,
 with

$\theta'(\bm{k},0)=\begin{cases}
      \theta_0 e^{i\alpha(\bm{k})} &(\delta k\leq|\bm{k}|\leq3\delta k,
      \\ 0 & \text{otherwise},
      \\
    \end{cases}$ 
       
where $\theta'(\bm{k},0)$ is the Fourier transform of $\theta(\bm{r},0)$, $\theta'(\bm{k},0)$ =  $\theta'^*(\bm{-k},0)$, $\alpha(\bm{k})$ is chosen randomly for each $\bm{k}$ in interval $[-\pi, \pi]$ and $\delta k = 2\pi/L$. 
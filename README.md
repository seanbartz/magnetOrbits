# magnetOrbits — scattering and capture of two free magnetic spheres

Two identical, uniformly magnetized spheres are projected toward each other with impact
parameter $b$. Both translate and both rotate freely in 3D (spin angular momentum
$I\boldsymbol\omega_i$ exchanges with orbital angular momentum). There is no contact force:
a trajectory that reaches $r=2R$ counts as a **collision** and stops there.

Background: [Bartz & Shaw, arXiv:2212.00876](https://arxiv.org/abs/2212.00876) (PDF and
LaTeX in `paper/`). The earlier attempt at this project is kept in `archive/`, for
reference only.

The staged plan and its validation status are in [PLAN.md](PLAN.md).

## Dimensionless variables

Scales: length $R$, energy $U_0=\mu_0\mu^2/4\pi R^3$, mass $m/2$,
time $t_0=\sqrt{(m/2)R^2/U_0}$. With $\mathbf r=\mathbf r_2-\mathbf r_1$:

$$\ddot{\mathbf r}=\mathbf F_2,\qquad \dot{\hat\mu}_i=\boldsymbol\omega_i\times\hat\mu_i,\qquad
\tilde I\,\dot{\boldsymbol\omega}_i=\boldsymbol\tau_i,\qquad \tilde I=2\beta=4/5,$$

$$U=-\frac{1}{r^3}\Big[3(\hat\mu_1\!\cdot\!\hat r)(\hat\mu_2\!\cdot\!\hat r)-\hat\mu_1\!\cdot\!\hat\mu_2\Big].$$

The inputs are $\tilde E$ (relative kinetic energy $/U_0$), $\tilde b=b/R$, and the initial
moment directions. For uniformly magnetized spheres,
$\tilde E=\tfrac34\rho v_{\rm rel}^2/\mu_0M^2$, independent of the sphere size.
Physical units appear only in `physics/model.py` (`scales`, `physical_to_dimensionless`).

## Program flow

```
physics/model.py        U, F_2, tau_1, tau_2                    (stage 1)
```

## Tests

```bash
python -m pytest -q
```

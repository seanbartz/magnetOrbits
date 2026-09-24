"""Published target: Bartz & Shaw, arXiv:2212.00876 (source in paper/source/arxivVersion.tex)."""

CITATION = ("S. P. Bartz and J. Shaw, 'Numerical simulation of non-central collisions of "
            "spherical magnets', arXiv:2212.00876")
SOURCE = ("Fig. 4 caption: 'The maximum value of b/R = 4.65, as the magnets do not collide for "
          "larger values at the initial velocity used in these simulations.'")

# Their parameters (Sec. 4.1): R/x_c = 6.94, each sphere moves at v_i = 5.78e-3 x_c/t_c, CM at rest,
# moments initially aligned with r_hat (head-to-tail), beta = 2/5, damped spring Gamma = 0.05.
PAPER_R_OVER_XC = 6.94
PAPER_SPEED_PER_SPHERE = 5.78e-3
PAPER_DAMPING = 0.05

CONVENTIONS = """
Their x_c = (3 mu0 mu^2 / 4 pi k)^(1/5), t_c = sqrt(m/k), so mu0 mu^2/(4 pi) = k x_c^5 / 3 and
our U0 = mu0 mu^2/(4 pi R^3) = k x_c^2 / (3 R~^3). Relative KE = (1/2)(m/2)(2 v_i)^2 = m v_i^2
= k x_c^2 v~^2. Hence E_tilde = 3 R~^3 v~^2 = 0.033501. Their b is the center-to-center lateral
offset (their Fig. 2), same as ours. Their orientation is 'aligned with r_hat' at an unstated
start separation; ours is head-to-tail along the incoming axis at t -> -inf (x_hat; the global
flip m -> -m makes -x_hat equivalent).
Definition mismatch: their figures include only runs that end STUCK as a barbell under the
damped spring; ours is 'reaches r = 2R'. Their integrator is Euler-Cromer with unstated dt.
"""

ENERGY_TILDE = 3 * PAPER_R_OVER_XC**3 * PAPER_SPEED_PER_SPHERE**2
TARGET_MAX_IMPACT_PARAMETER = 4.65
# Quoted to 3 significant figures: +- half a unit in the last digit. The spacing of their b grid
# is unstated, so the true uncertainty may be larger.
ABS_TOL = 0.005

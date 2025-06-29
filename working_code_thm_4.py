import numpy as np
from numpy.linalg import svd, norm
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Updated the output font for presentation purposes. Switched from sans serif to Computer Modern as used in LaTeX.
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10"]  # Use Computer Modern Roman as the serif font
})


def generate_near_symmetric_matrix(n, k, delta, spread_factor, seed=None):
    """
    We aim to make a near symmetric matrix, through mixing our U and V but a 'target orthogonal matrix'
    in order to make sure they are sufficiently close in structure. Q is made through skewing the identity.
    """
    if seed is not None:
        np.random.seed(seed)

    U_full, _ = np.linalg.qr(np.random.randn(n, n))
    V_full, _ = np.linalg.qr(np.random.randn(n, n))
    U = U_full[:, :k]
    V = V_full[:, :k]

    Q = np.eye(k)
    if delta > 0:
        perturbation = delta * np.random.randn(k, k)
        perturbation = (perturbation - perturbation.T) / 2
        Q = expm(perturbation)

    V_target = U @ Q
    t = 1 - delta
    V_new = t * V_target + (1 - t) * V
    V_new, _ = np.linalg.qr(V_new)

    singular_values = np.logspace(0, spread_factor, k)[::-1]  # spread control
    S = np.diag(singular_values)

    F = U @ S @ V_new.T
    actual_delta = min(norm(U.T @ V_new - Q_test, 2) for Q_test in [np.eye(k), -np.eye(k)])

    return F, U, S, V_new, actual_delta, Q


def construct_matrix_set(F, k, copies=5):
    """
    This functions outputs a toy set of matrices given by \Omega_{F,X}^{\epsilon}. We define the
    rank-k truncation of the SVD of F to be A, then AX = FX, and to increase the set, we append
    the lower rows with multiples of the upper rows of A to ensure rank(A) = k.
    """
    X = np.eye(F.shape[0], k)
    Uf, Sf, Vhf = svd(F, full_matrices=False)
    A = Uf[:, :k] @ np.diag(Sf[:k]) @ Vhf[:k, :]
    omega = [A.copy()]
    for i in range(1, copies):
        A_mod = A.copy()
        A_mod[:, -i] = A[:, 0]
        omega.append(A_mod)
    return omega, X, A


def calculate_upper_bound(A, F, X, epsilon, delta, omega_set):
    """
    This functionc calculates the upper bound of the theorem, 
    through calculating c as well as the norm of FX and the singular values of F.
    """
    k = np.linalg.matrix_rank(F)
    FX = F @ X

    U_F, S_F, V_Ft = svd(F, full_matrices=False)
    XtV = X.T @ V_Ft.T
    sing_vals = np.linalg.svdvals(XtV)
    sing_val_max = sing_vals[0]
    sing_val_min = sing_vals[-1]

    if sing_val_min < 1e-12:
        raise ValueError("Singular value too small")

    c = sing_val_max / (sing_val_min ** 2)
    supremum = max(norm(A - mat, 2) for mat in omega_set)
    factor = (c ** 2 * (np.sqrt(2 * epsilon) + np.sqrt(2 * delta))) / (1 - (np.sqrt(2 * epsilon) + np.sqrt(2 * delta)))

    validation = c * (np.sqrt(2 * epsilon) + np.sqrt(2 * delta))

    FX_norm = norm(FX, 2)
    bound = 4 * FX_norm * factor
    return supremum, bound, validation


# Loop the above functions for a singular value spread from 10^0 to 10^2
n, k = 50, 5
epsilon = 0.1
delta = epsilon # for simplicity
copies = 10

spread_factors = np.linspace(0, 2, 10)  # log10 spread from 10^0 to 10^2
suprema = []
bounds = []

for i, sf in enumerate(spread_factors):
    F, *_ = generate_near_symmetric_matrix(n, k, delta, spread_factor=sf, seed=42 + i)
    _, _, _, _, actual_delta, _ = generate_near_symmetric_matrix(n, k, delta, spread_factor=sf, seed=42 + i)
    omega_set, X, A = construct_matrix_set(F, k, copies)
    diff, bound, valid = calculate_upper_bound(A, F, X, epsilon, delta, omega_set)
    print("validation = ", valid)
    suprema.append(diff)
    bounds.append(bound)
    print("actual_delta = ",actual_delta)
    print("LHS supremum =", diff)

plt.figure(figsize=(10, 6))
plt.plot(spread_factors, suprema, label="Supremum (LHS)", marker='o')
plt.plot(spread_factors, bounds, label="Upper Bound (RHS)", marker='x')
plt.xlabel("Logarithmic Spread of Singular Values")
plt.ylabel("Spectral Norm Value")
plt.title(r'Supremum in $\Omega_{\mathcal{A},X}^{\epsilon}$ versus RHS bound')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

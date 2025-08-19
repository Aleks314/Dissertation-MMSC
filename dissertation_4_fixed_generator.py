import numpy as np
from numpy.linalg import svd, norm
from scipy.linalg import expm


def generate_near_symmetric_matrix(n, k, delta, seed=None):
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

    singular_values = np.sort(np.random.rand(k) * 10 + 1)[::-1]
    S = np.diag(singular_values)

    F = U @ S @ V_new.T

    actual_delta = min(norm(U.T @ V_new - Q_test, 2) for Q_test in [np.eye(k), -np.eye(k)])

    return F, U, S, V_new, actual_delta, Q


def construct_matrix_set(n, k, delta, copies=5, seed=None):
    F, U, S, V_new, actual_delta, Q = generate_near_symmetric_matrix(n, k, delta, seed)

    # Define X = I_{n,k}
    X = np.eye(n, k)

    # Truncate F to rank-k => A = F_k
    Uf, Sf, Vhf = svd(F, full_matrices=False)
    A = Uf[:, :k] @ np.diag(Sf[:k]) @ Vhf[:k, :]

    # Build Omega set: start with A, then clone and tweak a column
    omega = [A.copy()]
    for i in range(1, copies):
        A_mod = A.copy()
        A_mod[:, -i] = A[:, 0]  # duplicate first column into last-i-th
        omega.append(A_mod)

    # Verify all constraints
    for idx, mat in enumerate(omega):
        print(f"[{idx}] Rank: {np.linalg.matrix_rank(mat) == k}, AX=FX: {np.allclose(mat @ X, F @ X, atol=1e-8)}")

    return omega, F, X, A, actual_delta


# Usage Example
n = 50
k = 5
delta = 0.5
matrices, F, X, A, delta_actual = construct_matrix_set(n, k, delta, copies=10, seed=42)

print("Final delta:", delta_actual)

def calculate_upper_bound(A, F, X, epsilon, delta, omega_set):
    n = F.shape[0]
    k = np.linalg.matrix_rank(F)
    FX = F @ X
    X_pinv = np.linalg.pinv(X)

    # Compute condition parameter c
    U_F , S_F, V_Ft = np.linalg.svd(F, full_matrices=False)
    XtV = X.T @ V_Ft.T
    sing_vals = np.linalg.svdvals(XtV)
    sing_val_max = sing_vals[0]
    sing_val_min = sing_vals[-1]

    if sing_val_min < 1e-12:
        raise ValueError("Singular value too small, matrix X may be ill-conditioned")

    c = sing_val_max / (sing_val_min ** 2)

    # Supremum over omega
    diffs = [np.linalg.norm(A - matrix, 2) for matrix in omega_set]
    supremum = max(diffs)

    # Upper bound from Theorem 4
    factor = (c ** 2 * (np.sqrt(2 * epsilon) + np.sqrt(2 * delta))) / (1 - (np.sqrt(2 * epsilon) + np.sqrt(2 * delta)))
    FX_norm = np.linalg.norm(FX, 2)
    bound = 4 * FX_norm * factor

    return supremum, bound

if not matrices:
    raise RuntimeError("make_set_omega() returned an empty set. Try increasing epsilon or sample count.")

A_test = matrices[0]
diff, bound = calculate_upper_bound(A_test, F, X, epsilon=0.1, delta=0.1, omega_set=matrices)
print(f"Difference: {diff:.4f}, Bound: {bound:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv, norm, expm, orthogonal_procrustes

n = 5
k = 3
delta = 0.1

def generate_near_symmetric_matrix(n, k, delta, seed=None):
    """Generate a δ-near-symmetric rank-k matrix."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random orthogonal matrices
    U_full, _ = np.linalg.qr(np.random.randn(n, n))
    V_full, _ = np.linalg.qr(np.random.randn(n, n))
    
    U = U_full[:, :k]
    V = V_full[:, :k]

    # Create a rotation matrix Q that makes U and V δ-close
    # We'll use a small perturbation approach
    Q = np.eye(k)

    if delta > 0:
        # Add small random perturbation
        perturbation = delta * np.random.randn(k, k)
        perturbation = (perturbation - perturbation.T) / 2  # Make skew-symmetric
        Q = expm(perturbation)  # Orthogonal matrix via matrix exponential
    
    # Adjust V so that U^T V ≈ Q
    V_target = U @ Q
    # Blend between V_target and V to achieve desired delta
    t = 1 - delta  # Blending parameter
    V_new = t * V_target + (1 - t) * V
    V_new, _ = np.linalg.qr(V_new)  # Re-orthogonalize
    
    # Generate singular values
    singular_values = np.sort(np.random.rand(k) * 10 + 1)[::-1]
    S = np.diag(singular_values)
    
    # Create the matrix
    F = U @ S @ V_new.T
    
    orthog = print(Q@Q.T)

    # Verify it's δ-near-symmetric
    actual_delta = np.min([norm(U.T @ V_new - Q_test, 2) 
                          for Q_test in [np.eye(k), -np.eye(k)]])
    
    return F, U, S, V_new, actual_delta,  Q

# Call function and capture all outputs including Q
F_new, U_new, S_new, V_new, delta_new, Q = generate_near_symmetric_matrix(n, k, delta, seed=None)


print("delta = ",delta_new)
# Check orthogonality
print("\nQ shape:", Q.shape)
print("Q @ Q.T (should be identity):\n", Q @ Q.T)
print("Q.T @ Q (should also be identity):\n", Q.T @ Q)
print("Is Q orthogonal?", np.allclose(Q @ Q.T, np.eye(k)) and np.allclose(Q.T @ Q, np.eye(k)))

def make_set_omega(F, X, epsilon=0.3, num_samples=1000):
    FX = F @ X
    X_pinv = pinv(X)
    omega = []

    for _ in range(num_samples):
        A, _, _, _, _, _ = generate_near_symmetric_matrix(F.shape[0], np.linalg.matrix_rank(F), epsilon)
        A_proj = A - (A @ X - FX) @ X_pinv
        if np.linalg.matrix_rank(A_proj) == np.linalg.matrix_rank(F) and np.allclose(A_proj @ X, FX):
            omega.append(A_proj)

    return omega

matrices = make_set_omega(F_new, V_new)
print(matrices)

def calculate_upper_bound(A, F, X, epsilon, delta):

    n = F.shape[0]
    k = np.linalg.matrix_rank(F)
    FX = F @ X
    X_pinv = pinv(X)

    U_F , S_F, V_Ft = np.linalg.svd(F, full_matrices=False)
    XtV = X.T @ V_Ft.T
    sing_vals = np.linalg.svdvals(XtV)
    sing_val_max = sing_vals[0]
    sing_val_min = sing_vals[-1]

    c = (sing_val_max) / (sing_val_min)**2

    diffs = [np.linalg.norm(A - matrix, 2) for matrix in matrices]
    supremum = max(diffs)

    factor = (c ** 2 * (np.sqrt(2 * epsilon) + np.sqrt(2 * delta))) / (1 - (np.sqrt(2 * epsilon) + np.sqrt(2 * delta)))
    FX_norm = np.linalg.norm(FX, 2)
    bound = 4 * FX_norm * factor

    return supremum, bound

if not matrices:
    raise RuntimeError("make_set_omega() returned an empty set. Try increasing epsilon or sample count.")
A_test = matrices[0]
diff, bound = calculate_upper_bound(A_test, F_new, V_new, epsilon=0.1, delta=delta_new)
print(f"Difference: {diff:.4f}, Bound: {bound:.4f}")

















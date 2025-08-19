import numpy as np
from numpy.linalg import svd, matrix_rank, norm
from scipy.linalg import orthogonal_procrustes

np.random.seed(42)

def generate_near_symmetric_matrix(n, k, delta):
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    U, s, Vh = svd(A)
    s[k:] = 0
    S = np.diag(s)
    F = U @ S @ Vh
    F += delta * np.random.randn(n, n)
    F = 0.5 * (F + F.T)
    return F, U, np.diag(S), Vh.T, norm(F - F.T, 2)

def make_set_omega(F, X, epsilon=0.3, delta=1e-3, num_samples=10000):
    n = F.shape[0]
    k = X.shape[1]
    FX = F @ X
    matrices = []

    for i in range(num_samples):
        A = np.zeros((n, n))
        A[:, :k] = F[:, :k]  # Enforce AX = FX via X = I[:, :k]
        A[:, k:] = delta * np.random.randn(n, n - k)  # Fill rest

        U, s, Vh = svd(A)
        s[k:] = 0
        A = U @ np.diag(s) @ Vh

        AX_equals_FX = np.allclose(A @ X, FX, atol=1e-6)
        rank_ok = matrix_rank(A) == k

        U_k, _, V_k = svd(A, full_matrices=False)
        U_k = U_k[:, :k]
        V_k = V_k.T[:, :k]
        M = U_k.T @ V_k
        Q, _ = orthogonal_procrustes(M, np.eye(k))
        procrustes_ok = norm(M - Q, 2) <= epsilon
        dist = norm(M - Q, 2)

        if i % 1000 == 0:
            print(f"[{i}] Rank: {rank_ok}, AX=FX: {AX_equals_FX}, Procrustes: {procrustes_ok} (dist={dist:.4f})")

        if AX_equals_FX and rank_ok and procrustes_ok:
            matrices.append(A)

    return matrices

n = 50
k = 3
num_samples = 10000
epsilon = 0.5
delta = 0.01

F, U, S, V, actual_delta = generate_near_symmetric_matrix(n, k, delta)
print(f"F shape: {F.shape}, F rank: {matrix_rank(F)}, Actual delta: {actual_delta:.4f}")

X = np.eye(n)[:, :k]
matrices = make_set_omega(F, X, epsilon, delta, num_samples)

if not matrices:
    raise RuntimeError("make_set_omega() returned an empty set. Try increasing epsilon, delta, or num_samples.")

print(f"Generated {len(matrices)} matrices satisfying all constraints.")

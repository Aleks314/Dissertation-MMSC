import numpy as np
from scipy.linalg import svd, pinv, norm, orthogonal_procrustes

np.set_printoptions(precision=4, suppress=True)

def make_set_omega(F, X, epsilon=0.1, delta=0.1, eta=1e-2, num_samples=10000):
    n = F.shape[0]
    k = np.linalg.matrix_rank(F)
    FX = F @ X
    X_pinv = pinv(X)

    matrices = []
    for i in range(num_samples):
        # Add small perturbation to F
        noise = eta * np.random.randn(n, n)
        F_perturbed = F + noise

        # Truncated SVD to rank k
        U, S, Vh = svd(F_perturbed, full_matrices=False)
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]

        # Enforce affine constraint AX = FX
        A_proj = A_k - (A_k @ X - FX) @ X_pinv

        # Check constraints
        U_a, _, Vh_a = svd(A_proj, full_matrices=False)
        U_k = U_a[:, :k]
        V_k = Vh_a.T[:, :k]

        M = U_k.T @ V_k
        Q_opt, _ = orthogonal_procrustes(M, np.eye(k))

        satisfies_rank = np.linalg.matrix_rank(A_proj) == k
        satisfies_affine = np.allclose(A_proj @ X, FX, atol=1e-6)
        satisfies_procrustes = norm(M - Q_opt, 2) <= epsilon

        print(f"[{i}] Rank: {satisfies_rank}, AX=FX: {satisfies_affine}, Procrustes: {satisfies_procrustes}")

        if satisfies_rank and satisfies_affine and satisfies_procrustes:
            matrices.append(A_proj)

    return matrices

# Example usage to run this function
if __name__ == "__main__":
    n = 5
    k = 3
    delta = 0.1
    epsilon = 0.1

    np.random.seed(42)
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.sort(np.random.rand(k) * 10 + 1)[::-1])
    F = U[:, :k] @ S @ V[:, :k].T
    X = V[:, :k]

    matrices = make_set_omega(F, X, epsilon=epsilon, delta=delta, eta=1e-2, num_samples=10000)

    if not matrices:
        raise RuntimeError("make_set_omega() returned an empty set. Try increasing epsilon or sample count.")
    else:
        print(f"Success: {len(matrices)} matrices generated.")

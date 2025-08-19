import numpy as np
import matplotlib.pyplot as plt

#print("This is the new code")

test_matrix = np.array(
    [[ 18, -11,  1, -8,  6],
     [ 11,  -1, -6, -2,  9],
     [ -1,  -2,  7, -2, -3],
     [  3,  -9, 11, -5, -6],
     [-14,   9,  1,  6, -4]],            # <- keep integer type on load
    dtype=int
)

test_vector = np.array([1, 2, 3, 4, 5], dtype=int)

eigvals, eigvecs = np.linalg.eig(test_matrix)

product = test_matrix @ test_vector          # same as np.matmul

def gaussian_matrix(rows: int, cols: int, sigma: float) -> np.ndarray:
    """Return a rows×cols Gaussian kernel, normalised to sum to 1."""
    x, y = np.meshgrid(np.linspace(-1, 1, cols),
                       np.linspace(-1, 1, rows))
    d = np.sqrt(x**2 + y**2)
    g = np.exp(-(d**2) / (2.0 * sigma**2))
    return g / g.sum()

rows, columns, sigma = 5, 3, 1.0
random_matrix = gaussian_matrix(rows, columns, sigma)

X = np.array([[1, 2, 3],
              [6, 0, 8],
              [3, 2, 2],
              [6, 1, 3],
              [2, 3, 4]], dtype=int)

Y = test_matrix @ X

X_inv = np.linalg.pinv(X)         # Moore–Penrose pseudoinverse
DMD   = Y @ X_inv

def nystrom(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return Ny = B (Aᵀ B)⁺ Bᵀ   (all NumPy arrays)."""
    in_prod = A.T @ B
    return B @ (np.linalg.pinv(in_prod)) @ B.T

nystrom_output = nystrom(X, Y)

def my_silliness(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    B_inv = np.linalg.pinv(B)
    in_prod_AB = A.T @ B
    in_prod_BA = B @ A.T

    return B @ (np.linalg.pinv(in_prod_AB)) @ B.T @ np.linalg.pinv(in_prod_BA) @ B @ in_prod_AB @ B_inv

n = 10000
alpha = 0.01
U, _ = np.linalg.qr(np.random.randn(n, n))
lambdas = np.exp(-alpha * np.arange(n))
A_large = U @ np.diag(lambdas) @ U.T

c = 500
X_large = gaussian_matrix(n, c, sigma)

Y_large = A_large @ X_large

DMD   = Y_large @ np.linalg.pinv(X_large)
nystrom_output = nystrom(X_large, Y_large)
silliness = my_silliness(X_large,Y_large)

DMD_error_2 = np.linalg.norm(A_large - DMD, ord = 2) / np.linalg.norm(A_large, ord = 2)
Nystrom_error_2 = np.linalg.norm(A_large - nystrom_output, ord = 2) / np.linalg.norm(A_large, ord = 2)
Silliness_error_2 = np.linalg.norm(A_large - silliness, ord = 2) / np.linalg.norm(A_large, ord = 2)

DMD_error_F = np.linalg.norm(A_large - DMD, ord = 'fro') / np.linalg.norm(A_large, ord = 'fro')
Nystrom_error_F = np.linalg.norm(A_large - nystrom_output, ord = 'fro') / np.linalg.norm(A_large, ord = 'fro')
Silliness_error_F = np.linalg.norm(A_large - silliness, ord = 'fro') / np.linalg.norm(A_large, ord = 'fro')

print("eigenvalues  :", eigvals)
print("A·v          :", product, "\n")
print("Gaussian mat :\n", random_matrix, "\n")

print("Absolute error estimates:")
print("2 Norm: DMD ", DMD_error_2, "\n")
print("2 Norm: Nystorm ", Nystrom_error_2, "\n")
print("2 Norm: Silly ", Silliness_error_2, "\n")

print("Absolute error estimates:")
print("Frob Norm: DMD ", DMD_error_2, "\n")
print("Frobenius Norm: Nystrom ", Nystrom_error_2, "\n")
print("Frobenius Norm: Silly ", Silliness_error_2, "\n")

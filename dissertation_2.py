import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv, norm, expm
from itertools import product

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
    
    # Verify it's δ-near-symmetric
    actual_delta = np.min([norm(U.T @ V_new - Q_test, 2) 
                          for Q_test in [np.eye(k), -np.eye(k)]])
    
    return F, U, S, V_new, actual_delta

def compute_omega_epsilon_set(F, X, epsilon, num_samples=100):
    """
    Compute a sample of matrices from Ω^ε_{F,X}.
    Returns list of matrices A such that:
    - rank(A) = k
    - AX = FX
    - A is ε-near-symmetric
    """
    n = F.shape[0]
    k = np.linalg.matrix_rank(F)
    FX = F @ X
    
    # Compute the pseudoinverse for the constraint AX = FX
    X_pinv = pinv(X)
    
    # Get SVD of F for better initialization
    U_F, S_F, Vh_F = svd(F, full_matrices=False)
    V_F = Vh_F.T[:, :k]
    
    matrices = []
    
    for i in range(num_samples):
        # Strategy 1: Start with F itself (which satisfies AX = FX)
        if i < num_samples // 3:
            # Perturb F slightly to create variation
            perturbation = 0.1 * np.random.randn(n, n)
            perturbation = (perturbation + perturbation.T) / 2  # Make symmetric
            A_candidate = F + perturbation @ (np.eye(n) - X @ X_pinv)
        
        # Strategy 2: Create ε-near-symmetric matrix with same range as F
        elif i < 2 * num_samples // 3:
            # Generate new V that's ε-close to U
            V_new = U_F[:, :k] + epsilon * np.random.randn(n, k) * 0.5
            V_new, _ = np.linalg.qr(V_new)
            V_new = V_new[:, :k]
            
            # Create matrix with controlled near-symmetry
            S_new = S_F + 0.1 * np.random.randn(k) * np.diag(S_F)
            A_candidate = U_F[:, :k] @ np.diag(S_new) @ V_new.T
        
        # Strategy 3: Use the projection formula more carefully
        else:
            # Generate a matrix that's already ε-near-symmetric
            U_rand, _ = np.linalg.qr(np.random.randn(n, k))
            # Make V close to U
            V_rand = U_rand + epsilon * np.random.randn(n, k) * 0.3
            V_rand, _ = np.linalg.qr(V_rand)
            V_rand = V_rand[:, :k]
            
            S_rand = np.sort(np.random.rand(k) * np.max(S_F))[::-1]
            A_random = U_rand @ np.diag(S_rand) @ V_rand.T
            
            # Project to satisfy AX = FX
            A_candidate = FX @ X_pinv + (np.eye(n) - X @ X_pinv) @ A_random
        
        # Verify constraints
        # 1. Check rank
        rank_A = np.linalg.matrix_rank(A_candidate, tol=1e-10)
        if rank_A != k:
            continue
            
        # 2. Check AX = FX (should be satisfied by construction, but verify)
        if norm(A_candidate @ X - FX, 'fro') > 1e-10 * norm(FX, 'fro'):
            continue
        
        # 3. Check ε-near-symmetry
        try:
            U_A, S_A, Vh_A = svd(A_candidate, full_matrices=False)
            if len(S_A) < k:
                continue
                
            V_A = Vh_A.T[:, :k]
            U_A = U_A[:, :k]
            
            # Check near-symmetry
            Q_options = [np.eye(k), -np.eye(k)]
            # Also try some rotations
            for angle in [0, np.pi/4, np.pi/2]:
                c, s = np.cos(angle), np.sin(angle)
                if k == 2:
                    Q_options.append(np.array([[c, -s], [s, c]]))
            
            min_dist = min([norm(U_A.T @ V_A - Q, 2) for Q in Q_options])
            
            if min_dist <= epsilon * 1.5:  # Give some tolerance
                matrices.append(A_candidate)
                
        except Exception as e:
            continue
    
    return matrices

def compute_upper_bound(F, X, epsilon, delta):
    """Compute the upper bound from Theorem 4."""
    n, s = X.shape
    k = np.linalg.matrix_rank(F)
    
    # Get SVD of F
    U0, S0, Vh0 = svd(F, full_matrices=False)
    V0 = Vh0.T[:, :k]
    
    # Compute c = σ_max(X^T V_0) / σ_min(X^T V_0)^2
    XTV0 = X.T @ V0
    singular_values = svd(XTV0, compute_uv=False)
    c = singular_values[0] / (singular_values[-1] ** 2)
    
    # Check condition
    condition = c * (np.sqrt(2 * epsilon) + np.sqrt(2 * delta))
    if condition >= 1:
        return np.inf, c, condition
    
    # Compute bound
    FX_norm = norm(F @ X, 2)
    numerator = c**2 * (np.sqrt(2 * epsilon) + np.sqrt(2 * delta))
    denominator = 1 - condition
    bound = 4 * FX_norm * (numerator / denominator)
    
    return bound, c, condition

def test_theorem4_bound(n=20, k=5, s=8, delta=0.01, epsilon=0.03, num_trials=20):
    """Test whether the bound from Theorem 4 is tight."""
    
    results = {
        'actual_diameter': [],
        'upper_bound': [],
        'ratio': [],
        'c_values': [],
        'condition_values': []
    }
    
    successful_trials = 0
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Generate test matrix and sketch
        F, U_F, S_F, V_F, actual_delta = generate_near_symmetric_matrix(n, k, delta, seed=trial)
        
        # Strategy: Generate X that aligns well with V_F to get better condition number
        # Start with random matrix
        X_random, _ = np.linalg.qr(np.random.randn(n, s))
        
        # Mix with V_F to improve alignment and thus reduce c
        if s >= k:
            # Include some columns from V_F to ensure good conditioning
            X = np.hstack([V_F[:, :min(k, s//2)], X_random[:, :s-min(k, s//2)]])
            X, _ = np.linalg.qr(X)  # Re-orthogonalize
        else:
            X = X_random
        
        # Verify rank(FX) = k
        if np.linalg.matrix_rank(F @ X) != k:
            print(f"  Skipping: rank(FX) = {np.linalg.matrix_rank(F @ X)} != {k}")
            continue
        
        # Compute upper bound
        bound, c, condition = compute_upper_bound(F, X, epsilon, actual_delta)
        
        if np.isinf(bound):
            print(f"  Skipping: condition {condition:.3f} >= 1")
            # Try with even smaller epsilon/delta
            epsilon_adjusted = epsilon * 0.1
            delta_adjusted = actual_delta * 0.1
            bound, c, condition = compute_upper_bound(F, X, epsilon_adjusted, delta_adjusted)
            if np.isinf(bound):
                continue
        
        print(f"  Generating Ω^ε_{{F,X}} samples...")
        # Generate matrices from Ω^ε_{F,X}
        omega_matrices = compute_omega_epsilon_set(F, X, epsilon, num_samples=50)
        
        if len(omega_matrices) < 2:
            print(f"  Skipping: only {len(omega_matrices)} matrices found")
            continue
        
        # Compute actual diameter
        max_distance = 0
        for i in range(len(omega_matrices)):
            for j in range(i + 1, len(omega_matrices)):
                dist = norm(omega_matrices[i] - omega_matrices[j], 2)
                max_distance = max(max_distance, dist)
        
        # Store results
        results['actual_diameter'].append(max_distance)
        results['upper_bound'].append(bound)
        results['ratio'].append(bound / max_distance if max_distance > 0 else np.inf)
        results['c_values'].append(c)
        results['condition_values'].append(condition)
        
        successful_trials += 1
        print(f"  Actual diameter: {max_distance:.4f}")
        print(f"  Upper bound: {bound:.4f}")
        print(f"  Ratio (bound/actual): {bound/max_distance:.2f}")
        print(f"  c = {c:.4f}, condition = {condition:.4f}")
    
    print(f"\nSuccessful trials: {successful_trials}/{num_trials}")
    return results

def plot_results(results):
    """Plot the comparison between actual diameters and upper bounds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Actual vs Bound
    ax = axes[0, 0]
    actual = results['actual_diameter']
    bound = results['upper_bound']
    ax.scatter(actual, bound, alpha=0.6)
    
    # Add y=x line
    max_val = max(max(actual), max(bound))
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.set_xlabel('Actual Diameter')
    ax.set_ylabel('Upper Bound')
    ax.set_title('Upper Bound vs Actual Diameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ratio distribution
    ax = axes[0, 1]
    ratios = [r for r in results['ratio'] if r < 100]  # Filter out extreme values
    ax.hist(ratios, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Ratio (Bound / Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Bound Tightness')
    ax.axvline(x=1, color='r', linestyle='--', label='Tight bound')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: c values vs ratio
    ax = axes[1, 0]
    ax.scatter(results['c_values'], results['ratio'], alpha=0.6)
    ax.set_xlabel('c value')
    ax.set_ylabel('Ratio (Bound / Actual)')
    ax.set_title('Bound Tightness vs Condition Number c')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Condition vs ratio
    ax = axes[1, 1]
    ax.scatter(results['condition_values'], results['ratio'], alpha=0.6)
    ax.set_xlabel('Condition value')
    ax.set_ylabel('Ratio (Bound / Actual)')
    ax.set_title('Bound Tightness vs Condition')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_condition_number(n, k, s, num_samples=100):
    """Analyze the distribution of condition numbers c for random matrices and sketches."""
    c_values = []
    
    for _ in range(num_samples):
        # Generate random matrix
        F, U, S, V, _ = generate_near_symmetric_matrix(n, k, delta=0.01)
        
        # Generate random sketch
        X, _ = np.linalg.qr(np.random.randn(n, s))
        
        # Compute c
        XTV = X.T @ V
        try:
            sing_vals = svd(XTV, compute_uv=False)
            if sing_vals[-1] > 1e-10:  # Check for numerical stability
                c = sing_vals[0] / (sing_vals[-1]**2)
                c_values.append(c)
        except:
            continue
    
    return np.array(c_values)

# First analyze why c is typically large
print("\nAnalyzing condition number distribution...")
c_distribution = analyze_condition_number(n=15, k=5, s=8, num_samples=100)
if len(c_distribution) > 0:
    print(f"Condition number statistics:")
    print(f"  Mean c: {np.mean(c_distribution):.2f}")
    print(f"  Median c: {np.median(c_distribution):.2f}")
    print(f"  Min c: {np.min(c_distribution):.2f}")
    print(f"  Max c: {np.max(c_distribution):.2f}")
    
    # What values of epsilon and delta would work?
    for epsilon in [0.001, 0.01, 0.1]:
        for delta in [0.001, 0.01, 0.1]:
            condition_vals = c_distribution * (np.sqrt(2*epsilon) + np.sqrt(2*delta))
            frac_valid = np.mean(condition_vals < 1) * 100
            if frac_valid > 0:
                print(f"  ε={epsilon}, δ={delta}: {frac_valid:.1f}% of cases satisfy condition < 1")

def generate_controlled_sketch(V, s, alignment_factor=0.5):
    """Generate a sketch matrix X that has controlled alignment with V."""
    n, k = V.shape
    
    if s >= k:
        # Include some columns from V to ensure good conditioning
        num_aligned = int(k * alignment_factor)
        X_aligned = V[:, :num_aligned]
        
        # Generate random orthogonal complement
        X_random, _ = np.linalg.qr(np.random.randn(n, s - num_aligned))
        
        # Combine and re-orthogonalize
        X = np.hstack([X_aligned, X_random])
        X, _ = np.linalg.qr(X)
    else:
        # For s < k, create a mixture
        X = alignment_factor * V[:, :s] + (1 - alignment_factor) * np.random.randn(n, s)
        X, _ = np.linalg.qr(X)
    
    return X

# Main execution
if __name__ == "__main__":
    print("Testing Theorem 4 Upper Bound...")
    print("=" * 50)
    
    # Test with your original matrix if desired
    test_matrix = np.array(
        [[ 18, -11,  1, -8,  6],
         [ 11,  -1, -6, -2,  9],
         [ -1,  -2,  7, -2, -3],
         [  3,  -9, 11, -5, -6],
         [-14,   9,  1,  6, -4]],
        dtype=float
    )
    
    # First, analyze the test matrix
    print("\nAnalyzing provided test matrix:")
    U, S, Vh = svd(test_matrix)
    V = Vh.T
    k = np.linalg.matrix_rank(test_matrix)
    
    # Check near-symmetry
    min_dist = np.min([norm(U[:, :k].T @ V[:, :k] - np.eye(k), 2),
                      norm(U[:, :k].T @ V[:, :k] + np.eye(k), 2)])
    print(f"Matrix rank: {k}")
    print(f"Near-symmetry measure: {min_dist:.4f}")
    print(f"Singular values: {S[:k]}")
    
    # Run systematic tests
    print("\n" + "=" * 50)
    print("Running systematic tests...")
    
    # Try multiple parameter sets to find working conditions
    param_sets = [
        {'n': 20, 'k': 5, 's': 12, 'delta': 0.01, 'epsilon': 0.02},  # More columns, smaller epsilon/delta
        {'n': 15, 'k': 4, 's': 10, 'delta': 0.005, 'epsilon': 0.01},  # Even smaller
        {'n': 10, 'k': 3, 's': 8, 'delta': 0.001, 'epsilon': 0.005},  # Very small
    ]
    
    all_results = None
    for params in param_sets:
        print(f"\nTrying parameters: n={params['n']}, k={params['k']}, s={params['s']}, "
              f"δ={params['delta']}, ε={params['epsilon']}")
        
        results = test_theorem4_bound(**params, num_trials=15)
        
        if results['actual_diameter']:  # If we got any successful results
            all_results = results
            break
    
    if all_results is None:
        # If still no results, try with extremely small values
        print("\nTrying with extremely small δ and ε...")
        all_results = test_theorem4_bound(
            n=10, k=3, s=8, delta=0.0001, epsilon=0.0005, num_trials=20
        )
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    if all_results and all_results['actual_diameter']:
        results = all_results
        ratios = np.array(results['ratio'])
        valid_ratios = ratios[ratios < 100]  # Filter extreme values
        
        print(f"Average ratio (bound/actual): {np.mean(valid_ratios):.2f}")
        print(f"Median ratio: {np.median(valid_ratios):.2f}")
        print(f"Min ratio: {np.min(valid_ratios):.2f}")
        print(f"Max ratio: {np.max(valid_ratios):.2f}")
        
        # Plot results
        plot_results(results)
    else:
        print("No valid results obtained.")
    
    # Test specific case with controlled parameters
    print("\n" + "=" * 50)
    print("Testing with controlled sketch alignment...")
    
    # Create a well-conditioned test case
    n, k, s = 12, 4, 8
    F, U, S_diag, V, _ = generate_near_symmetric_matrix(n, k, delta=0.001)
    
    # Generate sketch with controlled alignment
    controlled_results = {
        'alignment': [],
        'c_values': [],
        'conditions': [],
        'bounds': [],
        'actual_diameters': [],
        'ratios': []
    }
    
    for alignment in np.linspace(0.3, 0.9, 7):
        X = generate_controlled_sketch(V, s, alignment_factor=alignment)
        
        # Check rank condition
        if np.linalg.matrix_rank(F @ X) == k:
            bound, c, condition = compute_upper_bound(F, X, epsilon=0.01, delta=0.001)
            
            controlled_results['alignment'].append(alignment)
            controlled_results['c_values'].append(c)
            controlled_results['conditions'].append(condition)
            controlled_results['bounds'].append(bound if not np.isinf(bound) else None)
            
            print(f"\nAlignment factor: {alignment:.2f}")
            print(f"  c = {c:.4f}")
            print(f"  condition = {condition:.4f}")
            print(f"  bound valid: {'Yes' if not np.isinf(bound) else 'No'}")
            
            # If bound is valid, try to compute actual diameter
            if not np.isinf(bound):
                print(f"  Computing actual diameter...")
                omega_matrices = compute_omega_epsilon_set(F, X, epsilon=0.01, num_samples=200)
                
                if len(omega_matrices) >= 2:
                    max_distance = 0
                    for i in range(len(omega_matrices)):
                        for j in range(i + 1, len(omega_matrices)):
                            dist = norm(omega_matrices[i] - omega_matrices[j], 2)
                            max_distance = max(max_distance, dist)
                    
                    controlled_results['actual_diameters'].append(max_distance)
                    ratio = bound / max_distance if max_distance > 0 else np.inf
                    controlled_results['ratios'].append(ratio)
                    
                    print(f"  Found {len(omega_matrices)} matrices in Ω^ε_{{F,X}}")
                    print(f"  Actual diameter: {max_distance:.4f}")
                    print(f"  Upper bound: {bound:.4f}")
                    print(f"  Ratio (bound/actual): {ratio:.2f}")
                else:
                    print(f"  Could only find {len(omega_matrices)} matrices in Ω^ε_{{F,X}}")
                    controlled_results['actual_diameters'].append(None)
                    controlled_results['ratios'].append(None)
            else:
                controlled_results['actual_diameters'].append(None)
                controlled_results['ratios'].append(None)
    
    # Plot the effect of alignment
    if controlled_results['alignment']:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(controlled_results['alignment'], controlled_results['c_values'], 'bo-')
        plt.xlabel('Alignment Factor')
        plt.ylabel('Condition Number c')
        plt.title('Effect of Sketch Alignment on c')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(controlled_results['alignment'], controlled_results['conditions'], 'ro-')
        plt.axhline(y=1, color='k', linestyle='--', label='Validity threshold')
        plt.xlabel('Alignment Factor')
        plt.ylabel('Condition Value')
        plt.title('Effect of Alignment on Bound Validity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot actual diameters vs bounds where available
        plt.subplot(2, 2, 3)
        valid_indices = [i for i, d in enumerate(controlled_results['actual_diameters']) if d is not None]
        if valid_indices:
            valid_alignments = [controlled_results['alignment'][i] for i in valid_indices]
            valid_bounds = [controlled_results['bounds'][i] for i in valid_indices]
            valid_diameters = [controlled_results['actual_diameters'][i] for i in valid_indices]
            
            plt.plot(valid_alignments, valid_bounds, 'go-', label='Upper Bound')
            plt.plot(valid_alignments, valid_diameters, 'mo-', label='Actual Diameter')
            plt.xlabel('Alignment Factor')
            plt.ylabel('Value')
            plt.title('Upper Bound vs Actual Diameter')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot ratios where available
        plt.subplot(2, 2, 4)
        valid_ratios = [r for r in controlled_results['ratios'] if r is not None]
        if valid_ratios:
            valid_ratio_alignments = [controlled_results['alignment'][i] 
                                     for i, r in enumerate(controlled_results['ratios']) if r is not None]
            plt.plot(valid_ratio_alignments, valid_ratios, 'ko-')
            plt.axhline(y=1, color='r', linestyle='--', label='Tight bound')
            plt.xlabel('Alignment Factor')
            plt.ylabel('Ratio (Bound / Actual)')
            plt.title('Bound Tightness vs Alignment')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary of findings
        print("\n" + "=" * 50)
        print("Summary of Controlled Alignment Results:")
        if valid_ratios:
            print(f"Average ratio (bound/actual): {np.mean(valid_ratios):.2f}")
            print(f"The bound is approximately {np.mean(valid_ratios):.1f}x larger than actual diameter")
            print(f"Best case ratio: {np.min(valid_ratios):.2f}")
            print(f"Worst case ratio: {np.max(valid_ratios):.2f}")
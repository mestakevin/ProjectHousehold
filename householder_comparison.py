import numpy as np
import matplotlib.pyplot as plt
import time

# Function to generate a random symmetric matrix
def generate_symmetric_matrix(n):
    """Generates a random symmetric matrix of size n x n."""
    random_matrix = np.random.randint(1, 100, size=(n, n))
    symmetric_matrix = (random_matrix + random_matrix.T) // 2  # Ensure symmetry
    return symmetric_matrix

# Function to enforce a tridiagonal form
def enforce_tridiagonal(matrix):
    """Converts a matrix to tridiagonal form."""
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                matrix[i, j] = 0.0
    return matrix

# Householder transformation
def householder_transformation(matrix):
    """Transforms a symmetric matrix to tridiagonal form."""
    n = matrix.shape[0]
    tridiagonal = matrix.copy()
    for k in range(n - 2):
        x = tridiagonal[k+1:, k]
        norm_x = np.linalg.norm(x)
        v = x.astype(float)
        v[0] += np.sign(x[0]) * norm_x
        v /= np.linalg.norm(v)
        Hk = np.eye(n)
        Hk[k+1:, k+1:] -= 2.0 * np.outer(v, v)
        tridiagonal = Hk @ tridiagonal @ Hk.T
    return enforce_tridiagonal(tridiagonal)

# Gaussian elimination to diagonalize a tridiagonal matrix
def gauss_elim(mat):
    """Performs Gaussian elimination to diagonalize a tridiagonal matrix."""
    n = len(mat[0])
    for i in range(n-1):
        c = -(mat[i+1, i] / mat[i, i])
        mat[i+1] += c * mat[i]
    for i in range(n-1, 0, -1):
        c = -(mat[i-1, i] / mat[i, i])
        mat[i-1] += c * mat[i]
    return mat

# Extract eigenvalues from a diagonalized matrix
def calculate_eigenvalues(diagonalized_matrix):
    """Extracts eigenvalues from a diagonalized matrix."""
    return np.diagonal(diagonalized_matrix)

# Custom method for eigenvalues
def custom_eigenvalue_method(matrix):
    """Custom method for computing eigenvalues."""
    tridiagonal_matrix = householder_transformation(matrix)  # Apply Householder transformation
    diagonalized_matrix = gauss_elim(tridiagonal_matrix)  # Perform Gaussian elimination
    if diagonalized_matrix is not None:
        return calculate_eigenvalues(diagonalized_matrix)
    return None

# Benchmark function
def benchmark_eigenvalue_methods(max_n, step=5):
    """Compares execution time for custom and NumPy eigenvalue methods."""
    sizes = list(range(5, max_n + 1, step))
    custom_times = []
    numpy_times = []

    for n in sizes:
        symmetric_matrix = generate_symmetric_matrix(n)
        
        # Measure time for custom method
        start_time = time.time()
        _ = custom_eigenvalue_method(symmetric_matrix)
        custom_times.append(time.time() - start_time)
        
        # Measure time for NumPy's method
        start_time = time.time()
        _ = np.linalg.eigvals(symmetric_matrix)
        numpy_times.append(time.time() - start_time)
    
    return sizes, custom_times, numpy_times

# Generate data for plotting
max_matrix_size = 1000  # Define the largest matrix size to test
matrix_step = 100 # Define step size for matrix growth
sizes, custom_times, numpy_times = benchmark_eigenvalue_methods(max_matrix_size, matrix_step)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sizes, custom_times, label="Custom Method", marker='o')
plt.plot(sizes, numpy_times, label="NumPy Method", marker='x')
plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (seconds)")
plt.title("Comparison of Eigenvalue Computation Times")
plt.legend()
plt.grid()
plt.show()

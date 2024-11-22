import numpy as np

# Set a seed for reproducibility in random number generation
np.random.seed(12345)

##----------------------------------------------##
def generate_symmetric_matrix(n):
    """
    Generates a random symmetric matrix of size n x n.

    Parameters:
        n (int): The size of the matrix.

    Returns:
        symmetric_matrix (ndarray): The generated symmetric matrix.
    """
    # Generate a random matrix of integers
    random_matrix = np.random.randint(1, 10, size=(n, n))
    
    # Make the matrix symmetric by averaging it with its transpose
    symmetric_matrix = (random_matrix + random_matrix.T) // 2  # Integer division for symmetric integer matrix
    
    return symmetric_matrix
##----------------------------------------------##
def householder_transformation(matrix):
    """
    Diagonalizes a symmetric matrix using Householder transformations.

    Parameters:
        matrix (ndarray): The symmetric matrix to be diagonalized.

    Returns:
        tridiagonal (ndarray): The tridiagonal matrix after applying Householder transformations.
    """
    # Ensure the input matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Matrix must be symmetric!")

    n = matrix.shape[0]  # Get the size of the matrix
    tridiagonal = matrix.copy()  # Create a copy to preserve the original matrix

    # Perform the Householder transformations
    for k in range(n - 2):
        # Extract the vector x from the column below the diagonal
        x = tridiagonal[k+1:, k]
        
        # Compute the norm of x and construct the Householder reflection vector
        norm_x = np.linalg.norm(x)
        v = x.astype(float)  # Convert x to float to prevent integer overflow
        v[0] += np.sign(x[0]) * norm_x  # Adjust the first component
        v /= np.linalg.norm(v)  # Normalize the reflection vector

        # Construct the Householder matrix
        Hk = np.eye(n)  # Start with the identity matrix
        Hk[k+1:, k+1:] -= 2.0 * np.outer(v, v)  # Apply the Householder transformation

        # Update the matrix with the transformation
        tridiagonal = Hk @ tridiagonal @ Hk.T

    # Ensure the matrix is tridiagonal
    return enforce_tridiagonal(tridiagonal)
##----------------------------------------------##
# Ensure the matrix is tridiagonal by zeroing out small values
def enforce_tridiagonal(matrix):
    """
    Converts a matrix to tridiagonal form by zeroing out elements
    that are not on the main diagonal or immediate off-diagonals.
    """
    n = matrix.shape[0]  # Get the size of the matrix
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:  # If not on the main or immediate off-diagonals
                matrix[i, j] = 0.0  # Set the element to zero
    return matrix
##----------------------------------------------##
def main():
    """
    Main function to demonstrate matrix tridiagonalization and diagonalization.
    """
    n = 5  # Size of the matrix
    print(f"{n} by {n} symmetric matrix")
    symmetric_matrix = generate_symmetric_matrix(n)  # Generate a symmetric matrix
    print(symmetric_matrix)
    print()

    print("Tridiagonalized Householder matrix")
    tridiagonal_matrix = householder_transformation(symmetric_matrix)  # Apply Householder transformation
    print(tridiagonal_matrix)
    print()
##----------------------------------------------##
main()

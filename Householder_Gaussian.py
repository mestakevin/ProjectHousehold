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
def gauss_elim(mat):
    """
    Performs Gaussian elimination to diagonalize a matrix.

    Parameters:
        mat (ndarray): The input matrix.

    Returns:
        None (prints the diagonalized matrix if successful).
    """
    n = len(mat[0])  # Get the size of the matrix
    i = 0  # Row index for forward elimination
    # Forward elimination process
    for current_row, next_row in zip(mat, mat[1:]):
        # Check if the current row equals the last row (degenerate case)
        if all(a == b for a, b in zip(current_row, mat[-1])):
            return None  # Terminate if a degenerate row is found
        else:
            c = -(next_row[i] / current_row[i])  # Compute the scaling factor
            row_b = next_row + (c * current_row)  # Update the next row
            mat[i+1] = row_b
        i += 1

    # Transpose the matrix for backward elimination
    tran_mat = mat.T
    j = 0  # Column index for backward elimination
    for current_row, next_row in zip(tran_mat, tran_mat[1:]):
        if all(a == b for a, b in zip(current_row, tran_mat[-1])):
            return None  # Terminate if a degenerate column is found
        else:
            c = -(next_row[j] / current_row[j])  # Compute the scaling factor
            row_b = next_row + (c * current_row)  # Update the next column
            tran_mat[j+1] = row_b
        j += 1
    
    # Set small values to zero for numerical stability
    for k in range(n):
        for f in range(n):
            if abs(tran_mat[k, f]) < 1e-10:
                tran_mat[k, f] = 0.0
    return tran_mat
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

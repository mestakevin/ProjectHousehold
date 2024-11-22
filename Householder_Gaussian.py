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

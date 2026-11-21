import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    arr = np.array(A)
    n, m = arr.shape
    result = np.empty((m, n), dtype=arr.dtype)
    for j in range(m):
        result[j, :] = arr[:, j]   # row j of Aᵀ is column j of A
    return result

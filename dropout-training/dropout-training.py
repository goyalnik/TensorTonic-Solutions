import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    arr = np.asarray(x, dtype=np.float64)

    if p == 0.0:
        dropout_pattern = np.ones_like(arr, dtype=np.float64)
        return arr.copy(), dropout_pattern

    if rng is not None:
        rand_vals = rng.random(arr.shape)
    else:
        rand_vals = np.random.random(arr.shape)

    scale = 1.0 / (1.0 - p)
    keep_mask = rand_vals >= p
    dropout_pattern = np.where(keep_mask, scale, 0.0)
    output = arr * dropout_pattern

    return output, dropout_pattern
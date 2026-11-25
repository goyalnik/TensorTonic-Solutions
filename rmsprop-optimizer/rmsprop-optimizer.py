import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Step 1: Update squared gradient accumulator
    new_s = beta * s + (1 - beta) * np.square(g)

    # Step 2: Update parameters
    new_w = w - (lr * g) / (np.sqrt(new_s) + eps)

    # Round to 6 decimal places
    new_w = np.round(new_w, 6)
    new_s = np.round(new_s, 6)

    return new_w.tolist(), new_s.tolist()
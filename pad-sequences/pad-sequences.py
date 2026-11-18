import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    if N == 0:
        return np.zeros((0, 0), dtype=int)

    if max_len is None:
        L = max(len(s) for s in seqs)
    else:
        L = max_len

    out = np.full((N, L), pad_value, dtype=int)
    for i, s in enumerate(seqs):
        k = min(len(s), L)
        if k > 0:
            out[i, :k] = s[:k]
    return out
from collections import Counter
import numpy as np


def compute_keep_probs(
    token_counts: Counter,
    word2idx: dict[str, int],
    total_tokens: int,
    threshold: float = 1e-4,
) -> np.ndarray:
    vocab_size = len(word2idx)
    counts = np.zeros(vocab_size, dtype=np.float32)

    for word, count in token_counts.items():
        counts[word2idx[word]] = count

    freqs = counts / float(total_tokens)

    keep_probs = np.ones(vocab_size, dtype=np.float32)
    mask = freqs > 0

    keep_probs[mask] = (
        (np.sqrt(freqs[mask] / threshold) + 1.0) * (threshold / freqs[mask])
    )

    return np.minimum(keep_probs, 1.0)


def subsample_sentence(
    sentence: list[int],
    keep_probs: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    return [idx for idx in sentence if rng.random() < keep_probs[idx]]
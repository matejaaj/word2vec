from collections import Counter
import numpy as np


class NegativeSampler:
    def __init__(
        self,
        token_counts: Counter,
        word2idx: dict[str, int],
        power: float = 0.75,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)

        self.vocab_size = len(word2idx)
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for word, count in token_counts.items():
            counts[word2idx[word]] = count

        probs = counts ** power
        self.probs = probs / probs.sum()

        self.table = self._build_table()
        self.pos = 0

    def _build_table(self) -> np.ndarray:
        return self.rng.choice(self.vocab_size, size=10_000_000, replace=True, p=self.probs)

    def sample(self, num_negatives: int, positive_idx: int) -> np.ndarray:
        negatives = []
        while len(negatives) < num_negatives:
            if self.pos >= len(self.table):
                self.table = self._build_table()
                self.pos = 0
            idx = int(self.table[self.pos])
            self.pos += 1
            if idx != positive_idx:
                negatives.append(idx)
        return np.array(negatives, dtype=np.int64)
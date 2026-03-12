import numpy as np

from subsampling import subsample_sentence
from model import SkipGramNegativeSampling
from negative_sampling import NegativeSampler


class Trainer:
    def __init__(
        self,
        model: SkipGramNegativeSampling,
        encoded_sentences: list[list[int]],
        window_size: int,
        negative_sampler: NegativeSampler,
        num_negatives: int,
        learning_rate: float = 0.025,
        num_epochs: int = 5,
        keep_probs: np.ndarray | None = None,
        seed: int = 42,
    ):
        self.model = model
        self.encoded_sentences = encoded_sentences
        self.window_size = window_size
        self.negative_sampler = negative_sampler
        self.num_negatives = num_negatives
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.keep_probs = keep_probs
        self.rng = np.random.default_rng(seed)

    def _estimate_total_pairs(self) -> int:
        total = 0
        for sentence in self.encoded_sentences:
            n = len(sentence)
            for i in range(n):
                left = max(0, i - self.window_size)
                right = min(n, i + self.window_size + 1)
                total += (right - left - 1)
        return total * self.num_epochs

    def train(self) -> list[float]:
        epoch_losses = []
        starting_lr = self.learning_rate
        min_lr = starting_lr * 1e-4

        estimated_total_pairs = self._estimate_total_pairs()
        global_step = 0

        for epoch in range(1, self.num_epochs + 1):
            order = self.rng.permutation(len(self.encoded_sentences))
            epoch_loss = 0.0
            num_pairs = 0

            for sent_idx in order:
                sentence = self.encoded_sentences[sent_idx]

                if self.keep_probs is not None:
                    sentence = subsample_sentence(sentence, self.keep_probs, self.rng)

                if len(sentence) < 2:
                    continue

                for i, center_idx in enumerate(sentence):
                    left = max(0, i - self.window_size)
                    right = min(len(sentence), i + self.window_size + 1)

                    for j in range(left, right):
                        if j == i:
                            continue

                        context_idx = sentence[j]
                        num_pairs += 1
                        global_step += 1

                        progress = global_step / estimated_total_pairs
                        lr = max(starting_lr * (1.0 - progress), min_lr)

                        negative_indices = self.negative_sampler.sample(
                            self.num_negatives, context_idx
                        )
                        loss = self.model.train_step(
                            center_idx, context_idx, negative_indices, lr
                        )
                        epoch_loss += loss

            avg_loss = epoch_loss / max(num_pairs, 1)
            epoch_losses.append(avg_loss)
            print(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"avg_loss: {avg_loss:.4f} - "
            )

        return epoch_losses
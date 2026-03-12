import numpy as np

class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        rng = np.random.default_rng(seed)
        limit = 0.5 / embedding_dim

        self.W_in = rng.uniform(
            low=-limit,
            high=limit,
            size=(vocab_size, embedding_dim),
        ).astype(np.float32)

        self.W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float32)


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))

    def train_step(
        self,
        center_idx: int,
        context_idx: int,
        negative_indices: np.ndarray,
        learning_rate: float,
    ) -> float:
        negative_indices = np.asarray(negative_indices, dtype=np.int64)

        v_c = self.W_in[center_idx]
        u_o = self.W_out[context_idx]
        U_neg = self.W_out[negative_indices]

        pos_score = np.dot(u_o, v_c)
        neg_scores = U_neg @ v_c

        loss = float(-np.log(self.sigmoid(pos_score)) + np.sum(-np.log(self.sigmoid(-neg_scores))))

        pos_coeff = self.sigmoid(pos_score) - 1.0
        neg_coeffs = self.sigmoid(neg_scores)

        grad_v_c = pos_coeff * u_o + np.sum(neg_coeffs[:, None] * U_neg, axis=0)
        grad_u_o = pos_coeff * v_c
        grad_U_neg = neg_coeffs[:, None] * v_c

        self.W_in[center_idx] -= learning_rate * grad_v_c
        self.W_out[context_idx] -= learning_rate * grad_u_o
        self.W_out[negative_indices] -= learning_rate * grad_U_neg

        return loss

    def get_input_embeddings(self) -> np.ndarray:
        return self.W_in
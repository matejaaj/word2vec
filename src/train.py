import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


from preprocessing import preprocess_corpus
from subsampling import compute_keep_probs
from negative_sampling import NegativeSampler
from model import SkipGramNegativeSampling
from trainer import Trainer



def main():
    data = preprocess_corpus("data/raw/wikitext-2.txt", min_count=5)

    total_tokens = sum(len(s) for s in data["encoded_sentences"])

    keep_probs = compute_keep_probs(
        token_counts=data["token_counts"],
        word2idx=data["word2idx"],
        total_tokens=total_tokens,
        threshold=1e-4,
    )

    sampler = NegativeSampler(
        token_counts=data["token_counts"],
        word2idx=data["word2idx"],
    )

    model = SkipGramNegativeSampling(
        vocab_size=len(data["word2idx"]),
        embedding_dim=100,
    )

    trainer = Trainer(
        model=model,
        encoded_sentences=data["encoded_sentences"],
        window_size=5,
        negative_sampler=sampler,
        num_negatives=5,
        learning_rate=0.025,
        num_epochs=5,
        keep_probs=keep_probs,
    )

    losses = trainer.train()


    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "embeddings.npy", model.get_input_embeddings())
    np.save(out / "epoch_losses.npy", np.array(losses))

    vocab = [data["idx2word"][i] for i in range(len(data["idx2word"]))]
    with open(out / "vocab.json", "w") as f:
        json.dump(vocab, f)


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_loss.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
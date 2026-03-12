from collections import Counter
from pathlib import Path
import re


def load_lines(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^=+\s.*\s=+$", line):
            continue
        lines.append(line)
    return lines


def normalize_line(line: str) -> str:
    line = line.lower()
    line = line.replace(" @-@ ", "-")
    line = line.replace(" @,@ ", ",")
    line = re.sub(r"<unk>", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)


def build_vocab(
    token_lists: list[list[str]],
    min_count: int,
) -> tuple[dict[str, int], dict[int, str], Counter]:
    counts = Counter()
    for tokens in token_lists:
        counts.update(tokens)

    counts = Counter({w: c for w, c in counts.items() if c >= min_count})
    vocab = sorted(counts.keys())

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return word2idx, idx2word, counts


def encode_tokens(tokens: list[str], word2idx: dict[str, int]) -> list[int]:
    return [word2idx[t] for t in tokens if t in word2idx]


def preprocess_corpus(path: str, min_count: int) -> dict:
    lines = load_lines(path)

    token_lists = []
    for line in lines:
        normalized = normalize_line(line)
        tokens = tokenize(normalized)
        if tokens:
            token_lists.append(tokens)

    word2idx, idx2word, token_counts = build_vocab(token_lists, min_count=min_count)

    encoded_sentences = [encode_tokens(t, word2idx) for t in token_lists]

    return {
        "encoded_sentences": encoded_sentences,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "token_counts": token_counts,
    }
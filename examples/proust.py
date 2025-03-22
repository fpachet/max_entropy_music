import logging
from pathlib import Path

from mem.training.generator import SequenceGenerator

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    s = Path("data/literature/combray-extract.txt").read_text()[:12]
    g = SequenceGenerator(s, k_max=10).train(max_iter=22)
    for _ in range(10):
        print("".join(g.sample_seq(100)))

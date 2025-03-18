from pathlib import Path

from mem.training.generator import SequenceGenerator

if __name__ == "__main__":
    s = Path("combray-extract.txt").read_text()
    g = SequenceGenerator(s, k_max=10, max_iter=100).train()
    for _ in range(10):
        print("".join(g.sample_seq(10)))

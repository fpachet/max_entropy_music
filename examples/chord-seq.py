"""
Dummy chord sequence generator.
"""

from pathlib import Path

from mem.training.generator import SequenceGenerator

if __name__ == "__main__":
    content = Path("data/chords/chord-seq.txt").read_text()
    content = content.replace("\n", ";")
    chord_labels = [label.strip() for label in content.split(";")]
    g = (
        SequenceGenerator(chord_labels[:10000], k_max=5, max_iter=100, fast=True)
        .print_info()
        .train()
    )
    for i in range(10):
        print(g.sample_seq(10))

"""
Dummy chord sequence generator.
"""

from pathlib import Path

from mem.training.generator import SequenceGenerator

from datasets import load_dataset

"""This dataset comes from:
@misc {oliver_holloway_2025,
	author       = { {Oliver Holloway} },
	title        = { lmd_chords (Revision 4d6815c) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/ohollo/lmd_chords },
	doi          = { 10.57967/hf/4219 },
	publisher    = { Hugging Face }
}
"""


def download_chord_dataset():
    ds = load_dataset("ohollo/lmd_chords")
    chords_list = ds["train"]["chords"]
    with open("./data/chords/chord_sequences.txt", "w") as file:
        for chord_seq in chords_list:
            # If chord_seq is a list, join the chords with a space
            chords = "; ".join(chord_seq["symbol"])
            file.write(chords + "\n")


if __name__ == "__main__":
    # download_chord_dataset()
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

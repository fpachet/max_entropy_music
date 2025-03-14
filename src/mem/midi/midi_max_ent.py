from pathlib import Path

from mem.algo.max_ent_np import MaxEnt
from mem.midi.midi_io import MidiPitchCorpus, save_midi


class MidiMaxEnt(MaxEnt):
    def __init__(self, midi_file: Path | str, kmax: int = 10):
        self.midi_file = Path(midi_file)
        self.midi_io = MidiPitchCorpus(midi_file, pitch_shifts=[0])
        super().__init__(self.midi_io.index_seq, q=self.midi_io.voc_size, kmax=kmax)

    def train(self, max_iter: int = 10000):
        # save_midi(
        #     self.sample_seq(length=100),
        #     "/Users/proy/Desktop/generated-max-ent-numpy-before-training.mid",
        # )
        super().train(max_iter=max_iter)

    def training_callback(self, params):
        # save_midi(
        #     self.sample_seq(length=100),
        #     f"./output-checkpoint-" f"{self.checkpoint_index}.mid",
        # )
        # print(f"Saved checkpoint #{self.checkpoint_index}")
        self.checkpoint_index += 1

    def sample_seq(self, length: int = 0, burn_in=1000):
        length = length or self.M
        index_seq = super().sample_index_seq(length=length, burn_in=burn_in)
        return self.midi_io.indices_to_notes(index_seq)

    def save_midi(self, output_file: str = "./output.midi"):
        note_seq = self.sample_seq()
        save_midi(note_seq, output_file)


if __name__ == "__main__":
    kmax = 15
    max_iter = 10
    L = 2000
    sampling_loops = 10 * L

    me = MidiMaxEnt("../../../data/midifiles/bach_violin_partita.mid", kmax=kmax)
    # me = MidiMaxEnt("../data/test_sequence_3notes.mid", kmax=10)
    me.train(max_iter=max_iter)
    seq = me.sample_seq(length=L, burn_in=sampling_loops)
    save_midi(seq, "/Users/proy/Desktop/generated-PR.mid")
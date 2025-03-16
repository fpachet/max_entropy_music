import time

import numpy as np
import mido
from scipy.optimize import minimize
import random

from mem.midi.midi import save_midi

"""
A working version of the Max Entropy paper.
Implementation follows the paper closely.
Contexts are created only once, partition functions computed only when necessary
negative_log_likelihood and gradient computed together (option jac=True in scipy.minimize),
to reduce the number of computations needed
"""


class MaxEntropyMelodyGenerator:
    def __init__(self, midi_file, Kmax=10, lambda_reg=1.0):
        self.midi_file = midi_file
        self.Kmax = Kmax
        self.lambda_reg = lambda_reg
        self.notes = self.extract_notes()
        self.note_set = list(set(self.notes))
        self.voc_size = len(self.note_set)
        self.note_to_idx = {note: i for i, note in enumerate(self.note_set)}
        self.idx_to_note = {i: note for note, i in self.note_to_idx.items()}
        self.seq = np.array([self.note_to_idx[note] for note in self.notes])
        self.all_contexts = np.empty(len(self.seq), dtype=object)
        for mu in range(len(self.seq)):
            self.all_contexts[mu] = self.build_context(self.seq, mu)
        self.all_partitions = np.zeros(len(self.seq))

    def extract_notes(self) -> list[int]:
        """Extracts MIDI note sequence from a MIDI file."""
        mid = mido.MidiFile(self.midi_file)
        notes = []
        if (len(mid.tracks)) == 1:
            track = mid.tracks[0]
        else:
            track = mid.tracks[1]
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)
        return notes

    # @profile
    def sum_energy_in_context(self, J, context, center):
        t0 = time.perf_counter_ns()
        # center is not necessarily the center of context
        energy = 0
        for k in range(self.Kmax):
            j_k = J[k]
            if -k - 1 in context:
                energy += j_k[context.get(-k - 1), center]
            if k + 1 in context:
                energy += j_k[center, context.get(k + 1)]
        return energy

    def compute_partition_function(self, h, J, context):
        # return np.exp([h[sigma] + self.sum_energy_in_context(J, context, sigma) for sigma in range(self.vocabulary_size())]).sum()
        z = 0
        for sigma in range(self.voc_size):
            energy = h[sigma] + self.sum_energy_in_context(J, context, sigma)
            z += np.exp(energy)
        return z

    # @profile
    def build_context(self, seq, i):
        M = len(seq)
        context = {k: seq[i + k] for k in range(self.Kmax + 1) if i + k < M}
        context.update({-k: seq[i - k] for k in range(self.Kmax + 1) if i - k >= 0})
        return context

    def negative_log_likelihood_and_gradient(self, params):
        voc_size = self.voc_size
        # unflatten h and J parameters
        h = params[:voc_size]
        J_flat = params[voc_size:]
        J = {
            k: J_flat[k * voc_size**2 : (k + 1) * voc_size**2].reshape(
                (voc_size, voc_size)
            )
            for k in range(self.Kmax)
        }
        # compute all partition functions for all mu to be used by obj function and gradient
        self.all_partitions = np.zeros(len(self.seq))
        self.compute_all_Z(J, h)
        result = self.negative_log_likelihood(h, J), self.gradient(h, J)
        return result

    def compute_all_Z(self, J, h):
        self.all_partitions = [
            self.compute_partition_function(h, J, self.all_contexts[mu])
            for mu in range(len(self.seq))
        ]

    def negative_log_likelihood(self, h, J):
        loss = 0
        M = len(self.seq)
        for i, s_0 in enumerate(self.seq):
            context = self.all_contexts[i]
            Z = self.all_partitions[i]
            energy = h[s_0] + self.sum_energy_in_context(J, context, s_0)
            energy -= np.log(Z)
            loss += energy
        loss *= -1 / M
        l1_reg = sum(np.abs(J[k]).sum() for k in range(self.Kmax))
        loss += (self.lambda_reg / M) * l1_reg
        print(f"{loss=}")
        return loss

    def gradient(self, h, J):
        grad_h = np.zeros_like(h)
        grad_J = {k: np.zeros_like(J[k]) for k in range(self.Kmax)}
        M = len(self.seq)
        # local fields
        for r in range(self.voc_size):
            sum_grad = 0
            for mu, s_0 in enumerate(self.seq):
                if r == s_0:
                    sum_grad += 1
                context = self.all_contexts[mu]
                Z = self.all_partitions[mu]
                expo = np.exp(h[r] + self.sum_energy_in_context(J, context, r))
                sum_grad -= expo / Z
            grad_h[r] = -sum_grad / M
        # J
        for k in range(self.Kmax):
            for r in range(self.voc_size):
                for r2 in range(self.voc_size):
                    prob = 0
                    for mu, s_0 in enumerate(self.seq):
                        context = self.all_contexts[mu]
                        Z = self.all_partitions[mu]
                        if mu - k - 1 >= 0 and r == self.seq[mu - k - 1]:
                            if r2 == s_0:
                                prob += 1
                            prob -= (
                                np.exp(
                                    h[r2] + self.sum_energy_in_context(J, context, r2)
                                )
                                / Z
                            )
                        if mu + k + 1 < M and r2 == self.seq[mu + k + 1]:
                            if r == s_0:
                                prob += 1
                            prob -= (
                                np.exp(h[r] + self.sum_energy_in_context(J, context, r))
                                / Z
                            )
                    grad_J[k][r][r2] = -prob / M + (self.lambda_reg / M) * np.abs(
                        J[k][r][r2]
                    )
        grad_J_flat = np.concatenate([grad_J[k].flatten() for k in range(self.Kmax)])
        return np.concatenate([grad_h, grad_J_flat])

    def train(self, max_iter=1000):
        voc_2 = self.voc_size**2
        params_init = np.zeros(self.voc_size + self.Kmax * voc_2)
        res = minimize(
            self.negative_log_likelihood_and_gradient,
            params_init,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter},
        )
        return res.x[: self.voc_size], {
            k: res.x[
                self.voc_size + k * voc_2 : self.voc_size + (k + 1) * voc_2
            ].reshape((self.voc_size, self.voc_size))
            for k in range(self.Kmax)
        }

    def generate_sequence(self, h, J, length=20, burn_in=1000):
        # generate sequence of note indexes
        sequence = [random.choice(self.seq) for _ in range(length)]
        for _ in range(burn_in):
            idx = random.randint(0, length - 1)
            context = self.build_context(sequence, idx)
            energies = np.zeros(self.voc_size)
            for i in range(self.voc_size):
                energies[i] = np.exp(h[i] + self.sum_energy_in_context(J, context, i))
            energies = energies / energies.sum()
            proposed_note = np.random.choice(range(self.voc_size), p=energies)
            sequence[idx] = proposed_note
        # build sequence of notes
        seq2 = self.generate_sequence_final_checks(sequence, h, J)
        result = [self.idx_to_note[i] for i in seq2]
        return result

    def generate_sequence_final_checks(self, sequence, h, J):
        for idx in range(len(sequence)):
            context = self.build_context(sequence, idx)
            energies = np.zeros(self.voc_size)
            for i in range(self.voc_size):
                energies[i] = np.exp(h[i] + self.sum_energy_in_context(J, context, i))
            energies = energies / energies.sum()
            best_note = np.where(energies == max(energies))[0][0]
            sequence[idx] = best_note
        return sequence


if __name__ == "__main__":
    # Utilisation
    # generator = MaxEntropyMelodyGenerator("../../../data/bach_partita_mono.midi", Kmax=10)
    # generator = MaxEntropyMelodyGenerator("../../../data/partita_violin.mid", Kmax=10)
    # generator = MaxEntropyMelodyGenerator("../../../data/prelude_c.mid", Kmax=10)
    generator = MaxEntropyMelodyGenerator(
        "../../../data/midi/bach_jesus_joy.mid", Kmax=10
    )
    # generator = MaxEntropyMelodyGenerator("../../../data//Just_Friends-_Pat_Martino_Solo.mid", Kmax=10)
    # [generator, h_opt, J_opt] = pickle.load(open("../../../data/partita_violin.p", "rb"))
    h_opt, J_opt = generator.train(max_iter=100)
    print(f"{h_opt=}")
    print(f"{J_opt=}")
    # pickle.dump([generator, h_opt, J_opt], open("../../../data/partita_violin.p", "wb"))
    generated_sequence = generator.generate_sequence(
        h_opt, J_opt, burn_in=4000, length=300
    )
    save_midi(generated_sequence, "../../../data/maxent-generated_melody.mid")

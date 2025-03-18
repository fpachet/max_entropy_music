import random
from typing import Self

import numpy as np
from scipy.optimize import minimize

from mem.midi.midi import save_midi, extract_pitches_from_midi
from mem.algo import NDArrayFloat

"""
A working version of the Max Entropy paper.
Implementation follows the paper closely.
Contexts are created only once, partition functions computed only when necessary
negative_log_likelihood and gradient computed together (option jac=True in scipy.minimize),
to reduce the number of computations needed
"""


class MaxEntropySlow:
    h: NDArrayFloat
    J: dict[int, NDArrayFloat]

    def __init__(
        self,
        index_training_seq: Sized[int],
        /,
        *,
        q: int,
        kmax=10,
        l=1.0,
    ):
        self.Kmax = kmax
        self.lambda_reg = l
        self.voc_size = q
        self.seq = np.array(index_training_seq)
        self.all_contexts = np.empty(len(self.seq), dtype=object)
        for mu in range(len(self.seq)):
            self.all_contexts[mu] = self.build_context(self.seq, mu)
        self.all_partitions = np.zeros(len(self.seq))

    # @profile
    def sum_energy_in_context(self, context, center):
        # center is not necessarily the center of context
        energy = 0
        for k in range(self.Kmax):
            j_k = self.J[k]
            if -k - 1 in context:
                energy += j_k[context.get(-k - 1), center]
            if k + 1 in context:
                energy += j_k[center, context.get(k + 1)]
        return energy

    def compute_partition_function(self, context):
        # return np.exp([h[sigma] + self.sum_energy_in_context(J, context, sigma) for sigma in range(self.vocabulary_size())]).sum()
        z = 0
        for sigma in range(self.voc_size):
            energy = self.h[sigma] + self.sum_energy_in_context(context, sigma)
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
        self.h = params[:voc_size]
        J_flat = params[voc_size:]
        self.J = {
            k: J_flat[k * voc_size**2 : (k + 1) * voc_size**2].reshape(
                (voc_size, voc_size)
            )
            for k in range(self.Kmax)
        }
        # compute all partition functions for all mu to be used by obj function and gradient
        self.all_partitions = np.zeros(len(self.seq))
        self.compute_all_z()
        return self.negative_log_likelihood(), self.gradient()

    def compute_all_z(self):
        self.all_partitions = [
            self.compute_partition_function(self.all_contexts[mu])
            for mu in range(len(self.seq))
        ]

    def negative_log_likelihood(self):
        loss = 0
        M = len(self.seq)
        for i, s_0 in enumerate(self.seq):
            context = self.all_contexts[i]
            Z = self.all_partitions[i]
            energy = self.h[s_0] + self.sum_energy_in_context(context, s_0)
            energy -= np.log(Z)
            loss += energy
        loss *= -1 / M
        l1_reg = sum(np.abs(self.J[k]).sum() for k in range(self.Kmax))
        loss += (self.lambda_reg / M) * l1_reg
        print(f"{loss=}")
        return loss

    def gradient(self):
        grad_h = np.zeros_like(self.h)
        grad_J = {k: np.zeros_like(self.J[k]) for k in range(self.Kmax)}
        M = len(self.seq)
        # local fields
        for r in range(self.voc_size):
            sum_grad = 0
            for mu, s_0 in enumerate(self.seq):
                if r == s_0:
                    sum_grad += 1
                context = self.all_contexts[mu]
                Z = self.all_partitions[mu]
                expo = np.exp(self.h[r] + self.sum_energy_in_context(context, r))
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
                                    self.h[r2] + self.sum_energy_in_context(context, r2)
                                )
                                / Z
                            )
                        if mu + k + 1 < M and r2 == self.seq[mu + k + 1]:
                            if r == s_0:
                                prob += 1
                            prob -= (
                                np.exp(
                                    self.h[r] + self.sum_energy_in_context(context, r)
                                )
                                / Z
                            )
                    grad_J[k][r][r2] = -prob / M + (self.lambda_reg / M) * np.abs(
                        self.J[k][r][r2]
                    )
        grad_J_flat = np.concatenate([grad_J[k].flatten() for k in range(self.Kmax)])
        return np.concatenate([grad_h, grad_J_flat])

    def train(self, max_iter=1000) -> Self:
        voc_2 = self.voc_size**2
        params_init = np.zeros(self.voc_size + self.Kmax * voc_2)
        res = minimize(
            self.negative_log_likelihood_and_gradient,
            params_init,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter},
        )
        self.h = res.x[: self.voc_size]
        self.J = {
            k: res.x[
                self.voc_size + k * voc_2 : self.voc_size + (k + 1) * voc_2
            ].reshape((self.voc_size, self.voc_size))
            for k in range(self.Kmax)
        }
        return self

    def sample_index_seq(self, length=20, burn_in=1000):
        # generate sequence of indices
        sequence = [random.choice(self.seq) for _ in range(length)]
        for _ in range(burn_in):
            idx = random.randint(0, length - 1)
            context = self.build_context(sequence, idx)
            energies = np.zeros(self.voc_size)
            for i in range(self.voc_size):
                energies[i] = np.exp(self.h[i] + self.sum_energy_in_context(context, i))
            energies = energies / energies.sum()
            proposed_note = np.random.choice(range(self.voc_size), p=energies)
            sequence[idx] = proposed_note
        return self.smooth_sequence(sequence)

    def smooth_sequence(self, sequence):
        for idx in range(len(sequence)):
            context = self.build_context(sequence, idx)
            energies = np.zeros(self.voc_size)
            for i in range(self.voc_size):
                energies[i] = np.exp(self.h[i] + self.sum_energy_in_context(context, i))
            energies = energies / energies.sum()
            best_note = np.where(energies == max(energies))[0][0]
            sequence[idx] = best_note
        return sequence


if __name__ == "__main__":
    # Utilisation
    # generator = MaxEntropyMelodyGenerator("../../../data/bach_partita_mono.midi", Kmax=10)
    # generator = MaxEntropyMelodyGenerator("../../../data/partita_violin.mid", Kmax=10)
    # generator = MaxEntropyMelodyGenerator("../../../data/prelude_c.mid", Kmax=10)
    generator = MaxEntropySlow("../../../data/midi/bach_jesus_joy.mid", Kmax=10)
    # generator = MaxEntropyMelodyGenerator("../../../data//Just_Friends-_Pat_Martino_Solo.mid", Kmax=10)
    # [generator, h_opt, J_opt] = pickle.load(open("../../../data/partita_violin.p", "rb"))
    generator.train(max_iter=100)
    # pickle.dump([generator, h_opt, J_opt], open("../../../data/partita_violin.p", "wb"))
    generated_sequence = generator.sample_index_seq(burn_in=4000, length=300)
    save_midi(generated_sequence, "../../../data/maxent-generated_melody.mid")

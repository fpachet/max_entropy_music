"""
Implementation of paper (see README.md), using numpy arrays whenever possible to
avoid Python loops. Compared to the slow implementation, this provides a significant
speedup to the price of more complex code.
"""

from __future__ import annotations

from typing import Sized

import numpy as np
import numpy.typing as npt
import tqdm
from scipy.optimize import minimize

from mem.algo import NDArrayInt, NDArrayFloat, FloatType, IdxType


class MaxEntFast:
    """
    A class to represent a Maximum Entropy model.

    Attributes:
        S: list[int] — 1D-numpy array of shape (M), S[µ] is the index
            of the symbol at position µ in the training sequence.

        M: int — the number of symbols in the training sequence; the typical index used
            to denote an index of an element in the training sequence is µ.

        q: int — the number of symbols in the vocabulary; the typical index used to
            denote an index of a symbol in the vocabulary is σ; if another symbol is
            needed, τ is used.

        K: int — the context width, i.e., the maximum distance from the center of
            a context; the typical index used to denote a distance from the center of a
            context is k.

        J: npt.NDArray[float] — 3D-numpy array of shape (K, q, q), element
            J(k, σ, τ) is the interaction potential between symbols σ and τ at
            distance k from each other within a context.

        h: np.ndarray — 1D-numpy array of shape (q), h[σ] is the bias of symbol σ.

        C: NDArrayInt — 2D-numpy array of shape (M, 2•kmax + 1), contexts[µ] is
            the context centered around the symbol at position µ.

        Z: npt.NDArray[float] — 1D-numpy array of shape (M), Z[µ] is the partition
            function for the context centered around the symbol at position µ.

        partition_ix: NDArrayInt — 4D-numpy array of shape (M, q, 3, 2•kmax),
            partition_ix[µ, σ] is the indices of the interaction potentials of the
            context centered around symbol index S[µ] with the center replaced by
            symbol index σ.

        J5: tuple[NDArrayInt, ...] — a tuple of 1D-numpy arrays of shape (M•2•kmax),
            J5 is created by reshaping context indices into a tuple of 3 1D-arrays;
            the first array contains distances to the center of a context; the second
            array is the index of a left symbol in a context; the third array is the
            index of a right symbol in a context; J5 is used to get in a single numpy
            access all the interaction potentials for all contexts in the training
            sequence using self.J[self.J5]; this is a lot faster than using for loops;
            this is used to compute Formula 5.

        J6: tuple[NDArrayInt, ...] — a tuple of 1D-numpy arrays of shape (M•2•kmax•q),
            J6 is created by reshaping partition_ix into a tuple of three 1D-arrays;
            the first array contains distances to the center of a context; the second
            array is the index of a left symbol in a context; the third array is the
            index of a right symbol in a context; J6 is used to get in a single numpy
            access all the interaction potentials for all contexts needed to compute
            Formula 6 using self.J[self.J6]; this is a lot faster than using for loops.

        J7: tuple[NDArrayInt, ...] — a tuple of 1D-numpy arrays of shape (q•M•2•kmax),
            J7 is created by reshaping partition_ix into a tuple of three 1D-arrays;
            the first array contains distances to the center of a context; the second
            array is the index of a left symbol in a context; the third array is the
            index of a right symbol in a context; J7 is used to get in a single numpy
            access all the interaction potentials for all contexts needed to compute
            Formula 7 using self.J[self.J7]; this is a lot faster than using for loops.

        K7: npt.NDArray[bool] — 2D-numpy array of shape (q, M), K7[σ, mu] is True if,
            and only if, S[µ] == σ; this is used to compute Formula 7.

        l: float — the regularization parameter, λ in the paper.
    """

    PADDING = -1

    S: list[int]
    M: int
    q: int
    K: int
    J: npt.NDArray[FloatType]
    h: np.ndarray
    C: NDArrayInt
    Z: npt.NDArray[FloatType]

    partition_ix: NDArrayInt
    J5: tuple[NDArrayInt, ...]
    J6: tuple[NDArrayInt, ...]
    J7: tuple[NDArrayInt, ...]

    K7: npt.NDArray[bool]

    l: float

    checkpoint_index: int

    def __init__(
        self,
        index_training_seq: Sized[int],
        *,
        q: int,
        kmax,
        l=1.0,
    ):
        self.S: NDArrayInt = np.array(index_training_seq, dtype=IdxType)
        self.M: int = len(self.S)
        self.q: int = q
        self.K: int = kmax
        self.l = l

        self.Z = np.zeros(self.M, dtype=FloatType)

        self.h = np.zeros(self.q, dtype=FloatType)

        # init J with j_init and an additional row of zeros at the end and an
        # additional column of zeros at the end of each row
        self.J = np.zeros((self.K, self.q + 1, self.q + 1), dtype=FloatType)

        self.C = compute_contexts(
            index_training_seq,
            kmax=self.K,
            padding=MaxEntFast.PADDING,
        )

        self.L_ix_arr = compute_context_indices(self.C, self.K)
        _indices = compute_context_indices(self.C, self.K)
        _indices = np.swapaxes(_indices, 0, 1)
        _indices = np.reshape(_indices, (3, -1))
        self.J5 = tuple(_indices)

        self.partition_ix = compute_partition_context_indices(
            self.C, q=self.q, kmax=self.K
        )  # shape is (M, q, 3, 2•K)
        _indices = np.swapaxes(self.partition_ix, 1, 2)  # (M, 3, q, 2•K)
        _indices = np.reshape(_indices, (self.M, 3, -1))  # (M, 3, 2•K•q)
        _indices = np.swapaxes(_indices, 0, 1)  # (3, M, 2•K•q)
        _indices = np.reshape(_indices, (3, -1))  # (3, M•2•K•q)
        self.J6 = tuple(_indices)

        _indices = np.swapaxes(self.partition_ix, 0, 1)  # (q, M, 3, 2•K)
        _indices = np.swapaxes(_indices, 1, 2)  # (q, 3, M, 2•K)
        _indices = np.swapaxes(_indices, 0, 1)  # (3, q, M, 2•K)
        _indices = np.reshape(_indices, (3, -1))  # (3, q•M•2•K)
        self.J7 = tuple(_indices)

        self.K7 = np.equal(np.arange(self.q)[:, None], self.S[None, :])

        # get the partition context indices and converts the 4D-array of shape
        # (M, q, 3, 2•kmax) into a list of length M, each element is a list of length q
        # whose elements are tuple of three 2•kmax numpy vectors This is to make
        # J[self.partition_context_indices[mu, sigma]] return the result using numpy
        # matrix indexing
        _partition_context_indices = compute_partition_context_indices(
            self.C, q=self.q, kmax=self.K
        )
        self.partition_context_indices = []
        for row_mu in _partition_context_indices:
            matrix_mu_sigma = []
            for col_sigma in row_mu:
                matrix_mu_sigma.append(tuple(col_sigma))
            self.partition_context_indices.append(matrix_mu_sigma)

        self.checkpoint_index = 0

    def save_checkpoint(self, path: str):
        np.savez(
            path,
            S=self.S,
            M=self.M,
            q=self.q,
            K=self.K,
            J=self.J,
            h=self.h,
            C=self.C,
            Z=self.Z,
            partition_ix=self.partition_ix,
            J5=self.J5,
            J6=self.J6,
            J7=self.J7,
            K7=self.K7,
            l=self.l,
            partition_context_indices=self.partition_context_indices,
            checkpoint_index=self.checkpoint_index,
        )

    @staticmethod
    def load_checkpoint(path: str) -> MaxEntFast:
        raise NotImplemented("Not implemented yet")

    # @Timeit
    def compute_z(self):
        """
        Compute the partition function Z for each context µ in the training sequence.

        Does it without using for loops by using numpy matrix indexing. See how
        self.J6 is laid out in the __init__ method for more details.

        Formula (6) in the referenced paper.
        """

        # all_j is essentially J[self.Z_ix] for all mu < self.M and all sigma < self.q
        # first, all_j is a 1D-array of shape (M * q * 2•kmax)
        all_j = self.J[self.J6]

        # then it is reshaped into a 3D-array of shape (M, q, 2•kmax)
        # so all_j[mu, sigma] is the array with all interaction potentials
        all_j = np.reshape(all_j, (self.M, self.q, -1))

        # concatenate the h vector to all the J vectors
        h = np.tile(self.h, (self.M, 1))
        h = h[:, :, None]
        # this is a 3D-array of shape (M, q, 2•kmax + 1)
        h_plus_all_j = np.concatenate((h, all_j), axis=2)

        # formula for Z
        self.Z = np.sum(np.exp(np.sum(h_plus_all_j, axis=2)), axis=1)

    def nll(self):
        """
        Compute the negative log-likelihood of the model given the training sequence.

        Formula (6) in the referenced paper.

        Returns:
            a float (the NLL)
        """
        sum_h = self.h[self.S].sum()
        sum_j = self.J[self.J5].sum()
        log_z = np.log(self.Z).sum()
        norm1_j = np.sum(np.abs(self.J))

        loss = (-(sum_h + sum_j - log_z) + self.l * norm1_j) / self.M
        print("loss={loss}".format(loss=loss))
        return loss

    def _grad_loc_field(self, _j_j7):
        """
        Formula (7) in the referenced paper.

        Returns:
            a 1D-numpy array of shape (q)
        """
        sum_potentials = np.sum(_j_j7.reshape(self.q, self.M, -1), axis=2)
        h_plus_sum_potentials = self.h[:, None] + sum_potentials
        return -(self.K7 - np.exp(h_plus_sum_potentials) / self.Z).sum(axis=1) / self.M

    def sum_kronecker_1(self, k):
        row_r = np.hstack(
            [
                np.full((self.q, k + 1), fill_value=False, dtype=bool),
                self.K7[:, : self.M - (k + 1)],
            ]
        )
        row_r2 = self.K7[:]
        return -np.count_nonzero(row_r.reshape(self.q, 1, self.M) * row_r2, axis=2)

    def sum_kronecker_2(self, k):
        row_r = self.K7[:]
        row_r2 = np.hstack(
            [
                self.K7[:, k + 1 :],
                np.full((self.q, k + 1), fill_value=False, dtype=bool),
            ]
        )
        return -np.count_nonzero(row_r.reshape(self.q, 1, self.M) * row_r2, axis=2)

    def exp1(self, _j_j7):
        kronecker = np.zeros((self.K, self.q, self.M), dtype=bool)
        for k in range(self.K):
            kronecker[k] = np.hstack(
                [
                    np.full((self.q, k + 1), fill_value=False, dtype=bool),
                    self.K7[:, : self.M - (k + 1)],
                ]
            )
        sum_potentials = np.sum(_j_j7.reshape(self.q, self.M, -1), axis=2)
        h_plus_sum_potentials = self.h[:, None] + sum_potentials
        normalized_exp = np.exp(h_plus_sum_potentials) / self.Z
        res = -np.sum(
            kronecker.reshape(self.K, self.q, 1, self.M)
            * normalized_exp.reshape(1, self.q, self.M),
            axis=3,
        )
        return res

    def exp2(self, _j_j7):
        kronecker = np.zeros((self.K, self.q, self.M), dtype=bool)
        for k in range(self.K):
            kronecker[k] = np.hstack(
                [
                    self.K7[:, k + 1 :],
                    np.full((self.q, k + 1), fill_value=False, dtype=bool),
                ]
            )
        sum_potentials = np.sum(_j_j7.reshape(self.q, self.M, -1), axis=2)
        h_plus_sum_potentials = self.h[:, None] + sum_potentials
        normalized_exp = np.exp(h_plus_sum_potentials) / self.Z
        res = -np.sum(
            kronecker.reshape(self.K, self.q, 1, self.M)
            * normalized_exp.reshape(1, self.q, self.M),
            axis=3,
        )
        return np.swapaxes(res, 1, 2)

    def regularization(self):
        return self.l * np.abs(self.J[:, : self.q, : self.q])

    def _grad_inter_pot(self, _j_j7):
        """
        Formula (8) in the referenced paper.
        Returns:
            a 1D-numpy array of shape (q)
        """
        dg_dj = -self.exp1(_j_j7)
        dg_dj -= self.exp2(_j_j7)

        for k in range(self.K):
            dg_dj[k] += self.sum_kronecker_1(k)
            dg_dj[k] += self.sum_kronecker_2(k)

        dg_dj += self.regularization()

        return dg_dj / self.M

    def update_arrays_from_params(self, params: NDArrayFloat):
        self.h = params[: self.q]
        self.J[:, : self.q, : self.q] = params[self.q :].reshape(self.K, self.q, self.q)

    def arrays_to_params(self):
        return np.concatenate([self.h, self.J[:, : self.q, : self.q].reshape(-1)])

    def nll_and_grad(self, params):
        self.update_arrays_from_params(params)
        self.compute_z()
        _j_j7 = self.J[self.J7]
        flat_grad = np.concatenate(
            [self._grad_loc_field(_j_j7), self._grad_inter_pot(_j_j7).reshape(-1)]
        )
        return self.nll(), flat_grad

    def training_callback(self, params):
        # self.save_checkpoint(f"./model-checkpoint-{self.checkpoint_index}")
        self.checkpoint_index += 1

    def train(self, max_iter=1000) -> Self:
        self.checkpoint_index = 0
        params_init = np.zeros(self.q + self.K * self.q * self.q)
        res = minimize(
            self.nll_and_grad,
            params_init,
            method="L-BFGS-B",
            # method="Powell",
            jac=True,
            callback=self.training_callback,
            options={"maxiter": max_iter},
        )
        self.h = res.x[: self.q]
        self.J[:, : self.q, : self.q] = res.x[self.q :].reshape(self.K, self.q, self.q)
        return self

    def sum_energy_in_context(
        self, seq: NDArrayInt, ix: int, center: int | None = None
    ):
        energy = 0
        for k in range(self.K):
            energy += self.J[k, seq[ix - (k + 1)], center]
            energy += self.J[k, center, seq[ix + (k + 1)]]
        return energy

    def sample_index_seq(
        self, length: int = 20, /, *, burn_in: int = 1000
    ) -> NDArrayInt:
        # generate sequence of note indexes
        index_seq = np.zeros(length + 2 * self.K, dtype=IdxType)
        index_seq[: self.K] = MaxEntFast.PADDING
        index_seq[-self.K :] = MaxEntFast.PADDING
        for _ in tqdm.tqdm(range(burn_in)):
            pos_in_seq = self.K + np.random.randint(0, length)
            energies = np.zeros(self.q)
            for new_center in range(self.q):
                energies[new_center] = np.exp(
                    self.h[new_center]
                    + self.sum_energy_in_context(index_seq, pos_in_seq, new_center)
                )
            energies = energies / energies.sum()
            proposed_note = np.random.choice(range(self.q), p=energies)
            index_seq[pos_in_seq] = proposed_note
        for _ in range(10):
            self.smooth_sequence(index_seq)
        return index_seq[self.K : -self.K]

    def smooth_sequence(self, seq):
        for idx in range(len(seq) - 2 * self.K):
            energies = np.zeros(self.q, dtype=FloatType)
            for i in range(self.q):
                energies[i] = np.exp(
                    self.h[i] + self.sum_energy_in_context(seq, idx + self.K, i)
                )
            energies = energies / energies.sum()
            best_note = np.where(energies == max(energies))[0][0]
            seq[idx + self.K] = best_note
        return seq


def compute_contexts(idx_seq: Sized[int], /, *, kmax: int, padding=-1) -> NDArrayInt:
    """
    Compute all contexts for a given index sequence.

    The input sequence contains indices of elements of the alphabet.
    Each context is a 2•kmax + 1 window around a sequence element.
    Note that the input sequence is left- and right-padded with kmax left_padding and
    right_padding indices respectively, to ensure that the first and last elements
    lead to contexts of length 2•kmax + 1.

    Args:
        idx_seq: the index sequence
        kmax: defines the size of contexts, i.e., 2•kmax + 1
        padding: the index used for padding

    Returns:
        a 2D-numpy array of shape (m, 2•kmax + 1) where m is the length of the
        input sequence

    """
    m = len(idx_seq)  # length of the input sequence
    l = 2 * kmax + 1  # length of the context
    c = np.zeros((m, l), dtype=int)  # the context matrix

    # left-pad and right-pad the input sequence
    padded_seq = np.concatenate(
        [np.array([padding] * kmax), idx_seq, np.array([padding] * kmax)]
    )

    for i in np.arange(m):
        c[i, :] = padded_seq[i : i + l]

    return c


def compute_context_indices(contexts: NDArrayInt, kmax: int = 0) -> NDArrayInt:
    """
    Compute the indices of each context in J, the 3D-array of interaction potentials.

    The result R is a 3D-array of shape (m, 3, 2•kmax). For a given context µ,
    R[µ] is a (3, 2•kmax) array such that J[R[µ]] is the interaction potential values
    corresponding to the context µ. If the context is [A, B, C, D, E, F, G] (assuming
    kmax = 3), then D is s_0 (the center of the context) and
        R[µ, 0] = [0, 0, 1, 1, 2, 2]
        R[µ, 1] = [C, D, B, D, A, D]
        R[µ, 2] = [D, E, D, F, D, G]

    Since J is indexed by K, left index, right index, J[R[µ]] will give the interaction
    potentials:
        J[0, C, D], J[0, D, E] // the potentials at distance 1 from the center
        J[1, B, D], J[1, D, F] // the potentials at distance 2 from the center
        J[2, A, D], J[2, D, G] // the potentials at distance 3 from the center

    Therefore, J[R[µ]].sum() will give the total interaction potential for the context
    µ, as computed in the Sum-Energy in Formula 5 in the paper (see README.md).

    Args:
        contexts: the 2D-array of contexts
        kmax: the maximum distance from the center of the context

    Returns:
        a 3D-array of shape (m, 3, 2•kmax)
    """
    m, ctx_length = contexts.shape
    kmax = kmax or (ctx_length - 1) // 2

    k_indices = np.tile(np.arange(kmax, dtype=int).repeat(2), m).reshape(m, -1)

    left_indices = np.zeros(2 * kmax, dtype=int)
    left_indices[0::2] = np.arange(kmax - 1, -1, -1)
    left_indices[1::2] = kmax

    right_indices = np.zeros(2 * kmax, dtype=int)
    right_indices[1::2] = np.arange(kmax + 1, 2 * kmax + 1)
    right_indices[0::2] = kmax

    left_indices = contexts[:, left_indices]
    right_indices = contexts[:, right_indices]

    indices = np.zeros((m, 3, 2 * kmax), dtype=int)

    indices[:, 0, :] = k_indices
    indices[:, 1, :] = left_indices
    indices[:, 2, :] = right_indices

    return indices


def compute_context_indices_naive(contexts, kmax=0):
    """
    same as compute_context_indices2 except much slower

    kept as a reference since it is easier to understand
    """
    m, l = contexts.shape
    kmax = kmax or (l - 1) // 2
    result = np.zeros((m, 3, 2 * kmax), dtype=int)
    for i in range(m):
        for k in range(kmax):
            s_0 = contexts[i, kmax]
            s_k = contexts[i, kmax + k + 1]
            s_mk = contexts[i, kmax - k - 1]
            result[i, 0, 2 * k : 2 * (k + 1)] = k
            result[i, 1, 2 * k] = s_mk
            result[i, 1, 2 * k + 1] = s_0
            result[i, 2, 2 * k] = s_0
            result[i, 2, 2 * k + 1] = s_k
    return result


def compute_partition_context_indices(
    contexts, /, *, q: int, kmax: int = 0
) -> NDArrayInt:
    """
    Compute the indices of each context in J with respect to each 0 ≤ sigma < q.

    The result R is a 4D-array of shape (m, q, 3, 2•kmax). For a given context µ, and
    an element index σ (0 ≤ σ < q), R[µ, σ] is a (3, 2•kmax) array such that J[R[µ, σ]]
    is the interaction potential values corresponding to the context µ if the
    context center is σ. If the context is [A, B, C, D, E, F, G] (assuming
    kmax = 3), then for a given σ:
        R[µ, σ, 0] = [0, 0, 1, 1, 2, 2]
        R[µ, σ, 1] = [C, σ, B, σ, A, σ]
        R[µ, σ, 2] = [σ, E, σ, F, σ, G]

    Since J is indexed by K, left index, right index, J[R[µ]] will give the interaction
    potentials:
        J[0, C, σ], J[0, σ, E]
        J[1, B, σ], J[1, σ, F]
        J[2, A, σ], J[2, σ, G]

    Therefore, J[R[µ, σ]].sum() will give the total interaction potential for the
    context µ and elements index σ, as computer in the Partition-Function, Formula (6)
    in the paper.

    Args:
        contexts: the 2D-array of contexts
        q: the size of the alphabet
        kmax: the maximum distance from the center of the context

    Returns:
        a 4D-array fo shape (m, q, 3, 2•kmax)
    """
    indices = compute_context_indices(contexts, kmax)
    normalization_indices = np.tile(
        indices.reshape(indices.shape[0], 1, *indices.shape[1:]), (1, q, 1, 1)
    )
    for sigma in range(q):
        normalization_indices[:, sigma, 1, 1::2] = sigma
        normalization_indices[:, sigma, 2, 0::2] = sigma

    return normalization_indices

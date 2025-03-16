from typing import Iterable, Any, TypeVar, Generic, Self

from mem.algo.max_entropy_fast import MaxEnt

T = TypeVar("T")


class SequenceGenerator(Generic[T]):
    training_sequence: tuple[T, ...]
    index_sequence: tuple[int, ...]
    vocabulary_size: int
    element_to_index: dict[T, int]
    index_to_element: dict[int, T]
    max_entropy: MaxEnt
    k_max: int

    def __init__(
        self, training_sequence: Iterable[T], /, *, k_max: int = 10, max_iter: int = 100
    ):
        self.training_sequence = tuple(training_sequence)
        _unique_elements = list(set(self.training_sequence))
        self.vocabulary_size = len(_unique_elements)
        self.index_to_element = dict(enumerate(_unique_elements))
        self.element_to_index = {v: k for k, v in self.index_to_element.items()}
        self.index_sequence = self.elements_to_indices(self.training_sequence)
        self.k_max = k_max
        self.max_iter = max_iter
        self.max_entropy = MaxEnt(
            self.index_sequence, q=self.vocabulary_size, kmax=self.k_max
        )

    def elements_to_indices(self, sequence: Iterable[T]) -> tuple[int, ...]:
        return tuple(self.element_to_index[char] for char in sequence)

    def indices_to_elements(self, indices: Iterable[int]) -> Iterable[T]:
        return tuple(self.index_to_element[idx] for idx in indices)

    def info(self) -> dict[str, Any]:
        return {
            "note_count": len(self.training_sequence),
            "voc_size": self.vocabulary_size,
            "note_seq": self.training_sequence,
            "index_seq": self.index_sequence,
        }

    def train(self) -> Self:
        self.max_entropy.train(max_iter=self.max_iter)
        return self

    def sample_seq(self, length: int, /, burn_in: int = 0) -> Iterable[T]:
        index_seq = self.max_entropy.sample_index_seq(
            length, burn_in=burn_in or 10 * length
        )
        return self.indices_to_elements(index_seq)

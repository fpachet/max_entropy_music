from typing import Iterable, Any, TypeVar, Generic, Self
from pathlib import Path
from mem.algo.max_entropy_fast import MaxEntropyFast
from mem.algo.max_entropy_slow import MaxEntropySlow

T = TypeVar("T")


class SequenceGenerator(Generic[T]):
    training_sequence: tuple[T, ...]
    __index_sequence: tuple[int, ...]
    vocabulary_size: int
    element_to_index: dict[T, int]
    index_to_element: dict[int, T]
    __max_entropy: MaxEntropyFast | MaxEntropySlow

    def __init__(
        self,
        training_sequence: Iterable[T],
        index_sequence: Iterable[int],
        vocabulary_size: int,
        element_to_index: dict[T, int],
        index_to_element: dict[int, T],
        max_entropy: MaxEntropyFast | MaxEntropySlow,
    ):
        self.training_sequence = tuple(training_sequence)
        self.__index_sequence = tuple(index_sequence)
        self.vocabulary_size = vocabulary_size
        self.element_to_index = dict(element_to_index)
        self.index_to_element = dict(index_to_element)
        self.__max_entropy = max_entropy

    @classmethod
    def on_sequence(
        cls, training_sequence: Iterable[T], /, *, k_max: int = 10, fast: bool = True
    ):
        training_sequence = tuple(training_sequence)
        _unique_elements = list(set(training_sequence))
        vocabulary_size = len(_unique_elements)
        index_to_element = dict(enumerate(_unique_elements))
        element_to_index = {v: k for k, v in index_to_element.items()}
        index_sequence = [element_to_index[x] for x in training_sequence]
        max_entropy = (MaxEntropyFast if fast else MaxEntropySlow).on_sequence(
            index_sequence, q=vocabulary_size, k_max=k_max
        )
        return cls(
            training_sequence,
            index_sequence,
            vocabulary_size,
            element_to_index,
            index_to_element,
            max_entropy,
        )

    @classmethod
    def train_on_sequence(
        cls,
        training_sequence: Iterable[T],
        /,
        *,
        k_max: int = 10,
        max_iter: int = 100,
        fast: bool = True,
    ):
        return cls.on_sequence(training_sequence, k_max=k_max, fast=fast).train(
            max_iter=max_iter
        )

    def set_model(self, max_entropy_model: MaxEntropyFast | MaxEntropySlow) -> None:
        self.__max_entropy = max_entropy_model

    def save_model(self, file_path: str | Path):
        self.__max_entropy.save_model(file_path)

    def load_model(self, file_path: str | Path) -> None:
        self.set_model(MaxEntropyFast.load_model(file_path))

    def elements_to_indices(self, sequence: Iterable[T]) -> tuple[int, ...]:
        return tuple(self.element_to_index[char] for char in sequence)

    def indices_to_elements(self, indices: Iterable[int]) -> Iterable[T]:
        return tuple(self.index_to_element[idx] for idx in indices)

    def print_info(self) -> Self:
        print(self.info())
        return self

    def info(self) -> dict[str, Any]:
        return {
            "note_count": len(self.training_sequence),
            "voc_size": self.vocabulary_size,
            "note_seq": self.training_sequence,
            "index_seq": self.__index_sequence,
        }

    def train(self, max_iter: int = 10) -> Self:
        self.__max_entropy.train(max_iter=max_iter)
        return self

    def sample_seq(self, length: int, /, burn_in: int = 0) -> Iterable[T]:
        index_seq = self.__max_entropy.sample_index_seq(
            length, burn_in=burn_in or 20 * length
        )
        return self.indices_to_elements(index_seq)

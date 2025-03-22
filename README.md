# Maximum Entropy Melody Generator

A melody generation system implementing a Maximum Entropy approach with the Ising/Potts model.
It is based on the paper:
Sakellariou, J., Tria, F., Loreto, V. et al. Maximum entropy models capture melodic styles. Sci Rep 7, 9172 (2017). https://doi.org/10. 
1038/s41598-017-08028-4 which is available [here](https://www.nature.com/articles/s41598-017-08028-4).

This model captures longer range interactions than Markov models (up to $K$) without the need for exponential amount of training data.
Number of parameters is $q + Kq^2$ where $q$ is the vocabulary size and $K$ is typically about 10.

## Authors
- [François Pachet](https://github.com/fpachet)
- [Pierre Roy](https://github.com/roypie)
 
## Implementation

There are two implementations of the model: a _pedagogical_ one, using Python loops, and a _fast_ one, using NumPy. The pedagogical version is useful to understand the model and the approach. The fast version is about 100 times faster and is useful for training and generating melodies.

The Numpy implementation references equations as they are numbered in [this paper](
https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-08028-4/MediaObjects/41598_2017_8028_MOESM49_ESM.pdf).

## Examples

The model is applicable to the generation of any type of sequences. It is trained on a single training sequence, such as melodies, chord sequences,
or character strings.

The `examples` directory contains examples for each type of sequences.

## Installation

The installation of the project is done using pip. The following steps will guide you through the installation process. It installs several Python 
packages, so it is best to create a virtual environment before installing the project.

1. Clone the repository:
```bash
git clone https://github.com/fpachet/max-entropy-music.git
cd max_entropy_music
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install the required dependencies:
```bash
pip install .
```

### Dependencies

The project requires the following Python packages:

    numpy ~= 2.2
    scipy ~= 1.15
    tqdm ~= 4.67
    mido ~= 1.3
    datasets ~= 3.4

## Usage

See Python files in the `examples` directory for examples of how to use the model.

The common approach to generate a sequence from a model trained on a training sequence is as follows:

1. Create a `SequenceGenerator` object from the sequence. You can optionally specify the context size of the model and select to use the slow, 
   pedagogical implementation if your goal is to see how the model is implemented (for instance using a step-by-step execution using a debugger).
2. Train the model by calling method `train` on the `SequenceGenerator` object. Note that the model works on a list of indexes, not on a list of 
   elements of the sequence. The `SequenceGenerator` object is a helper object that maps elements of the sequence to indexes and vice versa. So it 
   is a lot easier to use the sequence generator instead of a model.
3. Generate a new sequence by calling method `generate` on the `SequenceGenerator` object. You can specify the length of the generated sequence.

You can also save the trained model to a file and load it later from a `SequenceGenerator` object. In this case, you need not train the model 
again.  The example `examples/save_load_model.py` illustrate this.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
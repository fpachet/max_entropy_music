# Maximum Entropy Melody Generator

A melody generation system implementing a Maximum Entropy approach with the Ising/Potts model.
It is based on the paper:
Sakellariou, J., Tria, F., Loreto, V. et al. Maximum entropy models capture melodic styles. Sci Rep 7, 9172 (2017). https://doi.org/10.1038/s41598-017-08028-4

which is available at: https://www.nature.com/articles/s41598-017-08028-4

It captures longer range interactions than Markov models (up to K) without the need for exponential amount of training data.
Number of parameters is (vocabulary_size + K * vocabulary_size^2), with K typically about 10.

## Authors
- [François Pachet](https://github.com/fpachet)
- [Pierre Roy](https://github.com/roypie)
 
## Features

- Both a pedagogical and an efficient implementation of a Maximum Entropy model for melody generation.
The two implementations are equivalent. 
The pedagogical implementation is useful to understand the approach and the model.
The efficient implementation is about 100 X faster, and is useful for trying examples
- Examples on MIDI files and character sequences

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fpachet/max-entropy-music.git
cd max_entropy_music
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages:
numpy~=2.2.3
tqdm~=4.67.1
scipy~=1.15.2
mido~=1.3.3

## Usage

```python
from mem.algo import MaxEntropyModel
from utils.midi_processor import MIDIProcessor

# Initialize the model
model = MaxEntropyModel()

# Process MIDI data
processor = MIDIProcessor()
training_data = processor.load_midi_files('path/to/midi/files')

# Train the model
model.train(training_data)

# Generate new melody
new_melody = model.generate(length=32)

# Save the generated melody
processor.save_midi(new_melody, 'output.midi')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


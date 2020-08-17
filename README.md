# adversarial_signal_generation
Generative Adversarial Networks (GANs) have shown to produce excellent results in computer vision. In this repository it is demonstrated that authentic signals can also be generated with GANs. Sine curves of varying frequency, amplitude and offset are learned by recurrent neural networks.

# Requirements
- [x] python 3
- [x] tensorflow >= 2.0.0 (testet on version 2.3.0)
- [x] numpy

# Results
## Comparison Between Training Samples and Generated Samples
### Training Sines
<img src="https://github.com/janek-gross/adversarial_signal_generation/blob/master/training_data.png?raw=true" width="800" />

### Generated Sines
<img src="https://github.com/janek-gross/adversarial_signal_generation/blob/master/generated_data.png?raw=true" width="800" />

## Generalization to Longer Sequences
The following generated sequences are 4 times longer than the training sequences. Despite the rather short training the long sequences are often realistic.

<img src="https://github.com/janek-gross/adversarial_signal_generation/blob/master/generated_data_long.png?raw=true" width="800" />

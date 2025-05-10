This file is to drop random bits of knowledge/nomenclature/etc that is domain specific to neuroscience/digital signal process/the dataset as a whole.
Almost like an Appendix of all the things I'm learning as I do this project.

Power Spectral Density (PSD) Graph - 
A PSD graph visually represents the distribution of power within a signal across different frequencies. It's essentially a **frequency domain representation of a signal's energy**, showing which frequencies contribute most to the signal's power. X-axis = Hz. Y-axis represents the power spectral density, which quantifies the amount of power present at each frequency.

PCA vs ICA -
PCA aims to find uncorrelated principal components that capture the most variance in the data.
ICA focuses on finding statistically independent components that represent the original sources of the data. Assumes that the observed data is a linear combination of independent sources.
PCA aims to reduce dimensionality, while ICA aims to separate sources. 

Brain Wave Categories - 
The raw EEG has usually been described in terms of frequency bands: 
- Gamma (30-44Hz)
- BETA (13-30Hz)
- ALPHA (8-12 Hz)
- THETA (4-8 Hz)
- DELTA (less than 4 Hz)
Generally involved in deeply relaxed states

(Read more on brain waves here)[https://nhahealth.com/brainwaves-the-language/]
This file is to drop random bits of knowledge/nomenclature/etc that is domain specific to neuroscience/digital signal process/the dataset as a whole.
Basically a Glossary/Appendix of all the things I'm learning as I do this project.

Power Spectral Density (PSD) Graph - 
A PSD graph visually represents the distribution of power within a signal across different frequencies. It's essentially a **frequency domain representation of a signal's energy**, showing which frequencies contribute most to the signal's power. X-axis = Hz. Y-axis represents the power spectral density, which quantifies the amount of power present at each frequency.

PCA vs ICA -
PCA aims to find uncorrelated principal components that capture the most variance in the data.
ICA focuses on finding statistically independent components that represent the original sources of the data. Assumes that the observed data is a linear combination of independent sources.
PCA aims to reduce dimensionality, while ICA aims to separate sources. 

Brain Wave Categories - 
The raw EEG has usually been described in terms of frequency bands: 
- Gamma (30-45+Hz)
- BETA (13-30Hz)
- ALPHA (8-12 Hz)
- THETA (4-8 Hz)
- DELTA (less than 4 Hz)
Generally involved in deeply relaxed states

(Read more on brain waves here)[https://nhahealth.com/brainwaves-the-language/]

Preictal, Postictal, Interictal, Ictal -
Ictal is the period during a seizure.
interictal is the period that is not a seizure.
there is also pre and post ictal which is before and after a seizure respectively.

Mean Absolute Value (MAV) - 
The average magnitude of the signal in a given window.
Take an average of the absolute values for a given list

Bipolar Channels (IMPORTANT) -
Due to the setup of the sensors/EEG cap for this dataset, the channels are bipolar.
Meaning each channel reflects the difference between two scalp electrodes.

Absolute Band Power (ABP) - 
Frequency domain feature that is the integration of the PSD over a certain band of frequencies.
e.g. Alpha ABP would be the PSD integrated over 8-13Hz

Relative Band Power -
Like absolute band power except this value is 'normalized'.
Basically take ABP and divide it over the total power (which is ABP over 0.5-45Hz)

Spectral Entropy - 
a measure of the complexity of a spectrum, specifically how "random" or evenly distributed the spectral power is across different frequencies.
e.g. white noise (since it has a more even spread PSD) it will have a higher spectral entropy
[Read more here](https://dsp.stackexchange.com/questions/23689/what-is-spectral-entropy)
Basically, take a signal, convert to frequency domain, take the psd, 'normalize' the psd so that it's like a probability function, take entropy of that.
If every 'outcome' of that probability function is about the same, that results in high entropy.
Compared to a probability function with only a few highly probable outcomes, the entropy of that is lower.
- Usually spectral entropy is normalized to output a value between 0 and 1
    0 → signal has all energy concentrated in one frequency (low entropy).
    1 → signal has a flat (uniform) spectrum (maximum disorder/entropy).

Multiple ways of calculating PSD -
The ways of estimating PSD's have trade offs between accuracy, robustness, and computation time. from 'worst' to 'best'
- periodogram
fast, simple, but sensitive to noise. rarely used in research
- welch
balance between noise reduction & resolution. most common method in EEG feature extraction!
- multitaper
best spectral resolution & variance reduction. slower/complex. only should be used when fine/precise frequency content is needed (clinical research)





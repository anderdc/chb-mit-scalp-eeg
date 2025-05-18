'''
    This file contains various helper functions that facilitate data labeling, data processing, feature extraction
'''
import numpy as np
from scipy.signal import periodogram, welch
from scipy.integrate import simpson
from numpy.typing import NDArray

def label_epochs_by_annotations(epochs, raw, target_label="seizure") -> list[int]:
    """
    Labels epochs based on whether they overlap with annotations (like 'seizure').

    Parameters:
    - epochs: mne.Epochs object
    - raw: mne.Raw object with annotations
    - target_label: string that must appear in annotation descriptions to count as a match

    Returns:
    - list of binary labels (1 for seizure, 0 for non-seizure)
    """
    sfreq = raw.info["sfreq"]
    labels = []

    for ep in epochs:
        t0 = ep.events[0, 0] / sfreq
        t1 = t0 + epochs.tmax
        is_seizure = any(
            (ann["onset"] < t1) and (ann["onset"] + ann["duration"] > t0)
            for ann in raw.annotations if target_label.lower() in ann["description"].lower()
        )
        labels.append(int(is_seizure))

    return labels

def bandpower(
    data: NDArray[np.float64],
    fs: float,
    band: tuple[float, float],
    relative: bool = True,
    **kwargs,
) -> NDArray[np.float64]:
    """Compute the bandpower of the individual channels.

    Parameters
    ----------
    data : array of shape (n_channels, n_samples)
        Data on which the the bandpower is estimated.
    fs : float
        Sampling frequency in Hz.
    band : tuple of shape (2,)
        Frequency band of interest in Hz as 2 floats, e.g. ``(8, 13)``. The
        edges are included.
    relative : bool
        If True, the relative bandpower is returned instead of the absolute
        bandpower.
    **kwargs : dict
        Additional keyword arguments are provided to the power spectral density
        estimation function.
        The only provided arguments are the data array and the sampling
        frequency.

    Returns
    -------
    bandpower : array of shape (n_channels,)
        The bandpower of each channel.
    """
    assert data.ndim == 2, (
        "The provided data must be a 2D array of shape (n_channels, n_samples)."
    )
    # compute psd
    freqs, psd = welch(data, fs, **kwargs)

    assert len(band) == 2, "The 'band' argument must be a 2-length tuple."
    assert band[0] <= band[1], (
        "The 'band' argument must be defined as (low, high) (in Hz)."
    )
    # compute the bandpower
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])   # create a mask for the range of interest
    bandpower = simpson(psd[:, idx_band], dx=freq_res)              # integration
    bandpower = bandpower / simpson(psd, dx=freq_res) if relative else bandpower   # normalize (if computing relative band power)
    return bandpower

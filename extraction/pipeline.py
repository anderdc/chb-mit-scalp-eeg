def label_epochs_by_annotations(epochs, raw, target_label="seizure"):
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
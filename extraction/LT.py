import pandas as pd
import mne
import numpy as np

from extraction.tools import get_all_edf_files, get_seizure_path, get_all_seizures

WINDOW_SIZE = 2.5  # in seconds
OVERLAP = (0.5) * WINDOW_SIZE   # ALSO in seconds... doing it like this so I can just set as a % of window size

'''
    Load-Transform pipeline class that takes in list of EEG files to parse and will
    load:
        - annotate
        - segment
        - label
    transform:
        - feature extract
        - train/test split
    the data and store it all in a pandas dataframe 
'''
class LT:
    def __init__(self, file_names: list[str], window_size: int = WINDOW_SIZE, overlap: float = OVERLAP):
        """
            file_list: .edf file names to be parsed/included in the dataset
        """
        self.window_size = window_size
        self.overlap = overlap
        print(f"processing {len(file_list)} file(s)!")
        self.file_names = file_names
        
        self.seizures = get_all_seizures()
        
        # self.df = pd.DataFrame()

    def process(self, file_name: str):
        """
            Takes one file name, and runs the whole the pipeline on it, returns a dataframe
        """
        # pre-processing
        raw = self.annotate(file_name)
        
        raw.drop_channels(['--0', '--1', '--2', '--3', '--4', '--5'])   # dud channels
        raw.filter(l_freq=1, h_freq=50)
        
        X, y = self.segment(raw)

        # transform
        # TODO - you are here!
        # - extract all the features I used in feature_extraction
        # - make a function for each feature extracted
        # - call all functions
        # - put it all into a dataframe alongside the label vector
        
    def annotate(self, file_name) -> mne.io.Raw:
        """
            Takes a file name, reads its contents, converts to mne.io.raw, and annotates it if it contains any seizures
        """
        raw = mne.io.read_raw_edf(get_seizure_path(file_name), preload=True)
        
        # get the seizures for file, return if there are none
        seizures = [x for x in self.seizures if x.file_name == file_name]
        if(not len(seizures)):
            return raw

        # there are seizures, let's annotate
        onsets = []
        durations = []
        descriptions = []
        for seizure in seizures:
            onsets.append(seizure.start) 
            durations.append(seizure.end - seizure.start)
            descriptions.append("ictal")

        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(annotations)
        return raw

    def segment(self, raw: mne.io.Raw) -> tuple(np.ndarray, list[int]):
        """
            Takes an ANNOTATED mne.io.raw and segments it based on window size and overlap, returning an nd array
            Also generates a corresponding label vector for the segmented data
            returns:
                epoch data which is a (segments x channels x samples) dimensional array. aka (n_epochs, n_channels, n_times)
        """
        epochs = mne.make_fixed_length_epochs(raw, duration=self.window_size, overlap=self.overlap, preload=True)
        # print('total segments:', len(epochs))

        # Label Vector Generation
        sfreq = raw.info["sfreq"]   # sampling frequency
        labels = []
        
        for x in range(len(epochs.events[:, 0])):
            # using sampling freq, get start time and end time of a segment
            t0 = epochs.events[x, 0] / sfreq
            t1 = t0 + epochs.tmax
            # if there is ANY overlap of a seizure period and this segment, label = 1
            is_seizure = any(
                (ann["onset"] < t1) and (ann["onset"] + ann["duration"] > t0)
                for ann in raw.annotations if 'ictal' in ann["description"].lower()
            )
            labels.append(int(is_seizure))

        # return as 3 dimensional array. (segements x channels x samples). e.g. axis=2 runs along samples
        return (epochs.get_data(), labels)


    # def transform(self):
    # '''
    #     - reads seizure information for given file 
    #     - labels data
    #     - annotates data
    #     - segments the data
    #     - create label vector for data
    # '''
    #     pass


    # def extract_features(self):
    # '''
    #     uses prepped data to extract and append features to dataframe
    # '''
    #     pass


if __name__ == "__main__":
    # lt = LT()

    get_all_edf_files()


import pandas as pd
import mne
import numpy as np
import antropy as ent
from typing import Tuple, List

from extraction.tools import get_all_edf_files, get_seizure_path, get_all_seizures
from extraction.pipeline import bandpower

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
        print(f"processing {len(file_names)} file(s)!")
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

        channel_names = raw.describe(data_frame=True).name.tolist()
        print(f"channel names: {channel_names}")
        
        X, y = self.segment(raw)

        # transform
        # TODO - you are here!
        # - extract all the features I used in feature_extraction
        # - call all functions
        # - put it all into a dataframe alongside the label vector
        # df = pd.DataFrame(y, columns=['y'])

        # X_temp = self.extract_mean(X)

        # final_df = pd.concat([train_df.reset_index(drop=True), 
        #                      local_df["target"].reset_index(drop=True)], axis=1)
        # print(final_df.shape)

        
    def annotate(self, file_name) -> mne.io.Raw:
        """
            Takes a file name, reads its contents, converts to mne.io.raw, and annotates it if it contains any seizures
        """
        raw = mne.io.read_raw_edf(get_seizure_path(file_name), preload=True)
        
        # get the seizures for file, return if there are none
        seizures = [x for x in self.seizures if x['file_name'] == file_name]
        if(not len(seizures)):
            return raw

        # there are seizures, let's annotate
        onsets = []
        durations = []
        descriptions = []
        for seizure in seizures:
            onsets.append(seizure['start']) 
            durations.append(seizure['end'] - seizure['start'])
            descriptions.append("ictal")

        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(annotations)
        return raw

    def segment(self, raw: mne.io.Raw) -> Tuple[np.ndarray, List[int]]:
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

    '''
        Time Domain Features
    '''
    def extract_mean(self, data: np.ndarray) -> np.ndarray:
        """
            Takes an ndarray (segments x channels x samples)
            and returns an ndarray (segments x channels)
            samples are collapsed into one value per segment/epoch
        """
        return data.mean(axis=2)

    def extract_var(self, data: np.ndarray) -> np.ndarray:
        """
            see extract_mean description.
        """
        return data.var(axis=2)

    def extract_mav(self, data: np.ndarray) -> np.ndarray:
        """
            see extract_mean description.
        """
        return np.absolute(data).mean(axis=2)

    def extract_skewness(self, data: np.ndarray) -> np.ndarray:
        """
            see extract_mean description.
        """
        medians_per_epoch = np.median(data, axis = 2)
        std_per_epoch = data.std(axis = 2)
        return 3*(means_per_epoch - medians_per_epoch) / std_per_epoch

    '''
        Frequency Domain Features
    '''
    def _extract_band_power(self, band: Tuple[int, int], data: np.ndarray, sfreq: float, relative: bool) -> np.ndarray:
        """
            band - frequency band for which to calculate power
            data - segmented EEG
            sfreq - sampling frequency
            relative - whether to return relative band power or absolute band power
                relative band power is the ratio of power of a certain band against the power of the entire PSD (relative power / total power)
        """
        return np.array([
            bandpower(segment, sfreq, band, relative=relative)
            for segment in data
        ])

    def extract_delta_band_power(self, data: np.ndarray, sfreq: float, relative: bool) -> np.ndarray:
        band = (0.5, 4)
        return self._extract_band_power(band, data, sfreq, relative)

    def extract_theta_band_power(self, data: np.ndarray, sfreq: float, relative: bool) -> np.ndarray:
        band = (4, 8)
        return self._extract_band_power(band, data, sfreq, relative)

    def extract_alpha_band_power(self, data: np.ndarray, sfreq: float, relative: bool) -> np.ndarray:
        band = (8, 13)
        return self._extract_band_power(band, data, sfreq, relative)

    def extract_beta_band_power(self, data: np.ndarray, sfreq: float, relative: bool) -> np.ndarray:
        band = (13, 30)
        return self._extract_band_power(band, data, sfreq, relative)

    def extract_gamma_band_power(self, data: np.ndarray, sfreq: float, relative: bool) -> np.ndarray:
        band = (13, 30)
        return self._extract_band_power(band, data, sfreq, relative)

    def extract_spectral_entropy(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        return ent.spectral_entropy(data, sfreq, method='welch', normalize=True, axis=2)



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
    lt = LT(file_names=['chb15_06.edf'])

    lt.process('chb15_06.edf')
    


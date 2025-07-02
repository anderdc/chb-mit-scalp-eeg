import pandas as pd
import mne
import warnings
import numpy as np
import antropy as ent
import signal
import sys
import threading
from typing import Tuple, List, Dict
from functools import reduce
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

from extraction.tools import get_all_edf_files, get_seizure_path, get_all_seizures
from extraction.pipeline import bandpower
from extraction.logger import logger

THREAD_COUNT = 8   # controls how many files will be processed in parallel

WINDOW_SIZE = 2.5  # in seconds
OVERLAP = (0.5) * WINDOW_SIZE   # ALSO in seconds... doing it like this so I can just set as a % of window size

LOW_PASS_FILTER_FREQ = 50  # lowpass frequency for preprocessing data

'''
    Load-Transform pipeline class that takes in list of EEG files to parse and will
    load:
        - annotate
        - segment
        - label
    transform:
        - feature extract
        - standardize data to 0 mean unit variance
        - drop all features that are not common across all processed files
        - train/test split using leave one out patient validation
    the data and store it all in a pandas dataframe 
'''
class LTPipeline:
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    
    def __init__(self, file_names: list[str], verbose=False):
        """
            args:
                file_list - .edf file names to be parsed/included in the dataset
                verbose - if True, does more logging
        """
        self.verbose = verbose
        self.window_size = WINDOW_SIZE
        self.overlap = OVERLAP
        logger.info(f"{len(file_names)} total file(s) in pipeline!")
        self.file_names = file_names
        
        self.seizures = get_all_seizures()

        self.executor = None
        self.shutdown_event = threading.Event() 

    def train_test_split(self, validation_patient_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
            Does a train test split leaving one patient out as validation data. this is best for minimizing data leakage
            args:
                validation_patient_id - patient id to leave out for validation        
            returns: 
                tuple of pd.DataFrames - intended to be unpacked as X_train, X_test, y_train, y_test
        """
        # return if validation patient doesn't exist in data
        if validation_patient_id not in [x.split("_")[0] for x in self.file_names]:
            logger.error(f"Cannot find {validation_patient_id} in the processed data, did you include at least one file from that patient?")
            return
        if len(set([x.split("_")[0] for x in self.file_names])) == 1:
            logger.error("There is only one unique patient in passed in files. include at least two unique patients ")
            return 
        # run pipeline if not ran yet
        if not (self.X_train and self.X_test and self.y_train and self.y_test):
            data = self.run()
        else:
            return (self.X_train, self.X_test, self.y_train, self.y_test)

        # consolidate/split out train test
        X_train = pd.concat([df for patient_id, df in data.items() if patient_id != validation_patient_id], axis=0)
        y_train = X_train["label"]
        X_train.drop(columns=["label"], inplace=True)

        X_test = data[validation_patient_id]
        y_test = X_test["label"]
        X_test.drop(columns=["label"], inplace=True)

        # standardize the data, mean=0, std=1
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)   # only fit train data to avoid data leakage

        # convert scaled data to dataframes
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        # assign to object for easily grabbing values again
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        if self.verbose:
            print('*'*20, 'Train Test Split results', '*'*20) 
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Processes all files, using multithreading... extra logic added for properly closing spawned processes on keyboard interrupt
        saves as an object attribute, a dict where:
            key: patient_id
            value: pd.DataFrame
        """
        # helper function
        def signal_handler(signum, frame):
            print(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        
        patient_data = dict()
        
        def process_and_return(file: str):
            # Check if shutdown was requested
            if self.shutdown_event.is_set():
                return None
            patient_id = file.split("_")[0]
            df = self.process(file)
            return patient_id, df
        
        try:
            with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
                self.executor = executor
                futures = {executor.submit(process_and_return, file): file for file in self.file_names}
                
                # Process completed futures
                for future in as_completed(futures):
                    # Check if shutdown was requested
                    if self.shutdown_event.is_set():
                        break
                        
                    try:
                        result = future.result(timeout=1)  # Add timeout to prevent hanging
                        if result is None:  # Task was cancelled
                            continue
                            
                        patient_id, df = result
                        if patient_id not in patient_data:
                            patient_data[patient_id] = df
                        else:
                            patient_data[patient_id] = pd.concat([patient_data[patient_id], df], axis=0)
                            
                    except Exception as e:
                        logger.error(f"Failed processing {futures[future]}: {str(e)}")
                        
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down...")
            self.shutdown_event.set()
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            self.executor = None
            
        if not self.shutdown_event.is_set() and patient_data:
            # get common columns across all patient dataframes
            common_cols = list(reduce(lambda a, b: a & b, [set(df.columns) for df in patient_data.values()]))
            
            # prune non-common columns
            for patient_id, df in patient_data.items():
                patient_data[patient_id] = df[common_cols]
        
        return patient_data

    def process(self, file_name: str):
        """
            Takes one file name, and runs the the pipeline on it, returns a dataframe
        """
        if self.verbose:
            print("*"*60)
            logger.info(f"Processing file {file_name}")
        # pre-processing
        raw = self.annotate(file_name)

        channel_names = raw.describe(data_frame=True).name.tolist()
        sfreq = raw.info["sfreq"]   # sampling frequency
        if self.verbose:
            logger.info(f"{file_name} has {len(channel_names)} channels")

        # drop dud channels if any
        # TODO: do this somewhere else/to the file directly rather than here
        if '--0' in channel_names:
            raw.drop_channels(['--0']) 
        if '--1' in channel_names:
            raw.drop_channels(['--1'])
        if '--2' in channel_names:
            raw.drop_channels(['--2']) 
        if '--3' in channel_names:
            raw.drop_channels(['--3']) 
        if '--4' in channel_names:
            raw.drop_channels(['--4']) 
        if '--5' in channel_names:
            raw.drop_channels(['--5']) 
        
        raw.filter(l_freq=1, h_freq=LOW_PASS_FILTER_FREQ, verbose=False)

        X, y = self.segment(raw)
        
        # extraction
        df = self.transform(sfreq, channel_names, X, y)
    
        return df
    
    def _get_raw(self, file_name) -> mne.io.Raw:
        """
            helper function that gets the mne.io.Raw representation of an edf file
        """
        # ignoring warnings bc some files have duplicate channel names OR dud channels e.g. '--0', '--1'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return mne.io.read_raw_edf(get_seizure_path(file_name), preload=True, verbose=False)
        # return mne.io.read_raw_edf(get_seizure_path(file_name), preload=True, verbose=False)
           
    def annotate(self, file_name) -> mne.io.Raw:
        """
            Takes a file name, reads its contents, converts to mne.io.raw, and annotates it if it contains any seizures
        """
        raw = self._get_raw(file_name)
        
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

    def segment(self, raw: mne.io.Raw) -> Tuple[np.ndarray, np.ndarray]:
        """
            Takes an ANNOTATED mne.io.raw and segments it based on window size and overlap, returning an nd array
            Also generates a corresponding label vector for the segmented data
            returns:
                epoch data which is a (segments x channels x samples) dimensional array. aka (n_epochs, n_channels, n_times)
        """
        if self.verbose:
            logger.info("Segmenting...")
        epochs = mne.make_fixed_length_epochs(raw, duration=self.window_size, overlap=self.overlap, preload=True, verbose=False)
        if self.verbose:
            logger.info(f'total segments: {len(epochs)}')

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

        # return as 3 dimensional array. (segments x channels x samples). e.g. axis=2 runs along samples
        return (epochs.get_data(), np.array(labels))

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
        medians_per_epoch = np.median(data, axis=2)
        means_per_epoch = data.mean(axis=2)
        std_per_epoch = data.std(axis=2)
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

    def _to_df(self, feature: np.ndarray, prefix: str, channel_names: List[str]) -> pd.DataFrame:
        """
            takes an extracted feature of shape (n_segments, n_channels) and returns it as as a dataframe.
            prefix is prepended to the column names of the resulting dataframe.
            channel_names so that it can be part of the column names in resulting dataframe
        """
        n_channels = feature.shape[1]
        return pd.DataFrame(feature, columns=[f"{prefix}_{channel_names[i]}" for i in range(n_channels)])
    
    def transform(self, sfreq: str, channel_names: List[str], X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        '''
            extracts features, convert to dataframes, and concats all the data into one dataframe
        args:
            sfreq - sampling frequency of recorded EEG data
            channel_names - names of all electrodes
            X - segmented data of shape (n_segments, n_channels)
            y - target (label) vector
        returns:
            pd.DataFrame of all features, concatenated on axis=1
        '''        
        if self.verbose:
            logger.info(f"Now extracting features...")

        # time domain features
        mean_features = self.extract_mean(X)       # (n_segments, n_channels)
        var_features = self.extract_var(X)
        mav_features = self.extract_mav(X)
        skew_features = self.extract_skewness(X)
        
        # frequency domain features
        rel_delta_power = self.extract_delta_band_power(X, sfreq, relative=True)
        delta_power = self.extract_delta_band_power(X, sfreq, relative=False)
        
        rel_theta_power = self.extract_theta_band_power(X, sfreq, relative=True)
        theta_power = self.extract_theta_band_power(X, sfreq, relative=False)

        rel_alpha_power = self.extract_alpha_band_power(X, sfreq, relative=True)
        alpha_power = self.extract_alpha_band_power(X, sfreq, relative=False)

        rel_beta_power = self.extract_beta_band_power(X, sfreq, relative=True)
        beta_power = self.extract_beta_band_power(X, sfreq, relative=False)
        
        rel_gamma_power = self.extract_gamma_band_power(X, sfreq, relative=True)
        gamma_power = self.extract_gamma_band_power(X, sfreq, relative=False)

        entropy = self.extract_spectral_entropy(X, sfreq)

        # convert features to dataframes
        if self.verbose:
            logger.info(f"Converting to dataframe...")

        df_mean = self._to_df(mean_features, "mean", channel_names)
        df_var = self._to_df(var_features, "var", channel_names)
        df_mav = self._to_df(mav_features, "mav", channel_names)
        df_skew = self._to_df(skew_features, 'skew', channel_names)
        
        df_rel_delta_power = self._to_df(rel_delta_power, "d_pow_rel", channel_names)
        df_delta_power = self._to_df(delta_power, "d_pow", channel_names)
        df_rel_theta_power = self._to_df(rel_theta_power, "t_pow_rel", channel_names)
        df_theta_power = self._to_df(theta_power, "t_pow", channel_names)
        df_rel_alpha_power = self._to_df(rel_alpha_power, "a_pow_rel", channel_names)
        df_alpha_power = self._to_df(alpha_power, "a_pow", channel_names)
        df_rel_beta_power = self._to_df(rel_beta_power, "b_pow_rel", channel_names)
        df_beta_power = self._to_df(beta_power, "b_pow", channel_names)
        df_rel_gamma_power = self._to_df(rel_gamma_power, "g_pow_rel", channel_names)
        df_gamma_power = self._to_df(gamma_power, "g_pow", channel_names)
        df_entropy = self._to_df(entropy, "entropy", channel_names)

        # combine all features
        df = pd.concat([
            df_mean, df_var, df_mav, df_skew,
            df_rel_delta_power, df_delta_power, df_rel_theta_power, df_theta_power, df_rel_alpha_power, df_alpha_power, df_rel_beta_power, df_beta_power,
            df_rel_gamma_power, df_gamma_power, df_entropy
        ], axis=1)
        df["label"] = y

        if self.verbose: 
            logger.info(f"SUCCESS! Created dataframe with {df.shape[0]} segments (records) and {df.shape[1]} features.")
        return df

if __name__ == "__main__":
    ltp = LTPipeline(file_names=['chb15_06.edf', 'chb15_01.edf', 'chb01_01.edf', 'chb01_02.edf'])

    ltp.train_test_split('chb15')
    
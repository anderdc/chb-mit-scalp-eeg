

WINDOW_SIZE = 2.5  # in seconds
OVERLAP = (0.5) * WINDOW_SIZE   # ALSO in seconds... doing it like this so I can just set as a % of window size

'''
    Load-Transform pipeline class that takes in list of EEG files to parse and will
    - annotate
    - segment
    - label
    - feature extract
    - train/test split
    the data and store it all in a pandas dataframe 
'''
class LT:

    def __init__(self, ):
        pass

    def get_all_summary_data(self):
    '''
        loads up the seizure summary information .json file
        see parse_summaries.py for more information, you must run that script for the file to be created
    '''
        pass

    def load(self):
        pass


    def transform(self):
    '''
        - reads seizure information for given file 
        - labels data
        - annotates data
        - segments the data
        - create label vector for data
    '''
        pass


    def extract_features(self):
    '''
        uses prepped data to extract and append features to dataframe
    '''
        pass




import json
import os

def get_all_seizures():
    """
    Reads the all_summary_data.json file and returns a flattened list of seizures.
    
    Each seizure is represented as a dictionary with:
        - file_name: str
        - start: int
        - end: int

    The function assumes the JSON file is located at: <project_root>/data/all_summary_data.json.
    """
    # Determine where this script is located or imported from
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Build path to the data file
    data_path = os.path.join(project_root, '..', 'data', 'all_summary_data.json')

    # Normalize path (useful when importing across files)
    data_path = os.path.normpath(data_path)

    try:
    
        with open(data_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print('all_summary_data.json not file... have you parsed all summaries?')
        raise
    
    # Flatten the seizures
    flattened = []
    for record in all_data:
        file_name = record["file_name"]
        for seizure in record["seizure_times"]:
            flattened.append({
                "file_name": file_name,
                "start": seizure["start"],
                "end": seizure["end"]
            })
            
    return flattened

def get_seizure_path(filename: str):
    """
        given a filename to an .edf file, expand to get the full path from root of data directory
    """
    patient_id = filename.split("_")[0]
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Build path to the data file
    data_path = os.path.join(project_root, '..', 'data', patient_id, filename)

    return data_path




    
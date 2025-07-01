import json
import os

def get_all_edf_files() -> list[str]:
    """
        gets all file paths from data directory and return them as a list

        The function assumes the data directory is located at: <project_root>/data. (../data relative to this script)
    """
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, '..', 'data')

    all_files = list()

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.edf'):
                all_files.append(file)

    return all_files

def get_all_edf_files_for_patient(patient_id: str) -> list[str]:
    files = get_all_edf_files()
    return [file for file in files if file.startswith(patient_id)]

def get_all_seizures() -> list[dict]:
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

def get_seizure_path(filename: str) -> str:
    """
        given a filename to an .edf file, expand to get the full path from root of data directory
        tbh should just be named 'get_edf_path'
    """
    patient_id = filename.split("_")[0]
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Build path to the data file
    data_path = os.path.join(project_root, '..', 'data', patient_id, filename)

    return data_path




    